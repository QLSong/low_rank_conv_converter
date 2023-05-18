import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights


class LowRankConv2d(nn.Conv2d, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            r: int = 0,
            r_ratio: float = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
            merge_weights: bool = True,
            fix_low_rank: bool = False,
            fix_sparse: bool = False,
            tune_U: bool = False,
            tune_V: bool = False,
            tune_U_S: bool = False,
            tune_V_S: bool = False,
            keep_noise: bool = False,
            reshape_consecutive: bool = False,
            decompose_no_s: bool = False,
            tune_all: bool = False,
            lora_mode: bool = False,
            **kwargs
    ):
        self.reshape_consecutive = reshape_consecutive
        self.decompose_no_s = decompose_no_s
        if r == 0 and r_ratio > 0:
            sup_rank = min(in_channels * kernel_size, out_channels * kernel_size)
            r = min(int(in_channels * kernel_size * r_ratio), sup_rank)
            r = max(r, 1)

        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        assert type(kernel_size) is int
        # Actual trainable parameters
        if r > 0:
            # print("r is {}".format(r))
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_channels * kernel_size, r))
            )

            if not lora_mode:
                self.sparse_weight = nn.Parameter(
                    torch.zeros_like(self.weight)
                )
                self.sparse_mask = nn.Parameter(
                    torch.zeros_like(self.weight)
                )

            if keep_noise:
                self.noise = nn.Parameter(
                    torch.zeros_like(self.weight)
                )
                self.noise.requires_grad = False
            # self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            if not lora_mode:
                self.sparse_mask.requires_grad = False
            assert int(fix_low_rank) + int(fix_sparse) + int(tune_U) + int(tune_V) + int(tune_U_S) + int(tune_V_S) + int(tune_all) <= 1
            if fix_low_rank:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = False
                self.sparse_weight.requires_grad = True
            if fix_sparse:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = False
            if tune_U:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = False
            if tune_V:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = False
                self.sparse_weight.requires_grad = False
            if tune_U_S:
                self.lora_A.requires_grad = False
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = True
            if tune_V_S:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = False
                self.sparse_weight.requires_grad = True
            if tune_all:
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True
                self.sparse_weight.requires_grad = True

            self.keep_noise = keep_noise
            self.reset_parameters()

            self.lora_mode = lora_mode
            if self.lora_mode:
                assert not (keep_noise or self.reshape_consecutive)
                nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B)
                self.lora_A.requires_grad = True
                self.lora_B.requires_grad = True

    def reassign_sparsity(self, threshold):
        assert not self.lora_mode
        assert self.keep_noise
        assert self.r > 0
        residual = self.sparse_weight.data * (1 - self.sparse_mask.data) + self.noise.data
        self.sparse_mask.data = 0 * self.sparse_mask.data + (residual.abs() < threshold).float()
        self.sparse_weight.data = residual * (1 - self.sparse_mask.data)
        self.noise.data = residual - self.sparse_weight.data

    def merge(self, ):
        self.merged = True
        if self.lora_mode:
            self.weight_src = self.weight
            self.weight = self.lora_AB_to_weight() + self.weight_src
        else:
            self.weight = self.lora_AB_to_weight() + self.sparse_weight * (1 - self.sparse_mask.data)
            if self.keep_noise:
                self.weight.data += self.noise.data
        self.weight.requires_grad = True

    def lora_AB_to_weight(self):
        if self.reshape_consecutive:
            C_out, C_in, h, w = self.weight.shape
            consecutive_weight_shape = [C_out, h, w, C_in]
            weight = (self.lora_B @ self.lora_A).reshape(consecutive_weight_shape).permute(0, 3, 1, 2).contiguous()
        else:
            weight = (self.lora_B @ self.lora_A).view(self.weight.shape).contiguous()
        return weight

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            if self.lora_mode:
                weight = self.lora_AB_to_weight() + self.weight
            else:
                weight = self.lora_AB_to_weight() + self.sparse_weight * (1 - self.sparse_mask.data)
                if self.keep_noise:
                    weight += self.noise.data
            return F.conv2d(
                x, weight,
                self.bias, self.stride, self.padding, self.dilation, self.groups
            )
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def sparsity_regu(self, ):
        if hasattr(self, 'lora_A'):
            return torch.abs(self.sparse_weight).sum()
        else:
            return 0

    @torch.no_grad()
    def decompose(self, compress_step, lambda_s):
        if self.lora_mode:
            self.reset_parameters()
            return 1, 1

        residual_change = []
        sparse_weight_num = 0
        sparse_weight_all = 0

        if self.r > 0:
            if self.reshape_consecutive:
                # weight shape [C_out, C_in, h, w] -> [C_out, h, w, C_in]
                weight = self.weight.data.permute(0, 2, 3, 1).reshape(
                    self.lora_B.data.shape[0], self.lora_A.data.shape[1])
            else:
                weight = self.weight.data.reshape(self.lora_B.data.shape[0], self.lora_A.data.shape[1])

            U = torch.randn((self.lora_B.data.shape[0], 1), device=weight.device)
            V = torch.randn((1, self.lora_A.data.shape[1]), device=weight.device)

            for rank in range(self.r):
                S = torch.zeros_like(weight)

                for _ in range(compress_step):
                    U = torch.linalg.qr((weight - S) @ V.T)[0]
                    V = U.T @ (weight - S)
                    S = weight - U @ V
                    q = lambda_s
                    sparse_mask = (S.abs() < q)
                    if self.decompose_no_s:
                        sparse_mask = torch.ones_like(sparse_mask, device=sparse_mask.device)
                    S[sparse_mask] = 0
                residual_change.append(torch.norm(weight - U @ V).item() / torch.norm(weight))

                E = weight - U @ V - S
                E_vector = torch.linalg.qr(E)[1][:1]
                if (rank < self.r - 1):
                    V = torch.cat([V, E_vector])

            sparse_weight_num += int(sparse_mask.sum())
            sparse_weight_all += int(sparse_mask.numel())

            self.lora_B.data = 0 * self.lora_B.data + U.contiguous()
            self.lora_A.data = 0 * self.lora_A.data + V.contiguous()

            if self.reshape_consecutive:
                C_out, C_in, h, w = self.weight.shape
                consecutive_weight_shape = [C_out, h, w, C_in]
                self.sparse_weight.data = 0 * self.sparse_weight.data + S.reshape(consecutive_weight_shape).permute(0, 3, 1, 2).contiguous()
                self.sparse_mask.data = 0 * self.sparse_mask.data + sparse_mask.reshape(consecutive_weight_shape).permute(0, 3, 1, 2).float().contiguous()
                if self.keep_noise:
                    self.noise.data = 0 * self.noise.data + \
                                        E.reshape(consecutive_weight_shape).permute(0, 3, 1, 2).float().contiguous()
            else:
                self.sparse_weight.data = 0 * self.sparse_weight.data + S.reshape(self.sparse_weight.data.shape).contiguous()
                self.sparse_mask.data = 0 * self.sparse_mask.data + sparse_mask.reshape(self.sparse_weight.data.shape).float().contiguous()
                if self.keep_noise:
                    self.noise.data = 0 * self.noise.data + \
                                        E.reshape(self.sparse_weight.data.shape).float().contiguous()

        return sparse_weight_num, sparse_weight_all

def model_convert_conv(model, mode, path=None, convert_op_name=[], ignore_op_name=[], fix_op_name=[], r=0, r_ratio=0.25, compress_step=50, lambda_s=0.01, merge_parameter=False):
    assert mode in ['fix_sparse', 'fix_low_rank', 'tune_U', 'tune_V', 'tune_V_S', 'tune_U_S', 'tune_all', 'lora_mode']
    assert convert_op_name == [] or ignore_op_name == []
    args = {
        "fix_sparse" : False,
        "fix_low_rank" : False,
        "tune_U" : False,
        "tune_V" : False,
        "tune_V_S" : False,
        "tune_U_S" : False,
        "tune_all" : False,
        "lora_mode" : False,
        "keep_noise" : False,
        "reshape_consecutive" : False,
        "decompose_no_s" : False,
    }
    args[mode] = True
    if mode != 'lora_mode':
        assert path is not None

    convert_op_name_ = []
    ignore_op_name_ = []
    fix_op_name_ = []
    for name, op in model.named_modules():
        if type(op) == torch.nn.Conv2d:
            need_fix = False
            for op_name in fix_op_name:
                if op_name in name:
                    if hasattr(op, 'weight'):
                        op.weight.requires_grad = False
                    if hasattr(op, 'bias') and op.bias:
                        op.bias.requires_grad = False
                    fix_op_name_.append(name)
                    need_fix = True
            if need_fix:
                continue
            if op.groups > 1:
                ignore_op_name_.append(name)
                continue
            if len(convert_op_name) > 0:
                need_convert = False
                for op_name in convert_op_name:
                    if op_name in name:
                        need_convert = True
                if need_convert:
                    convert_op_name_.append(name)
                else:
                    ignore_op_name_.append(name)
            elif len(ignore_op_name) > 0:
                need_ignore = False
                for op_name in ignore_op_name:
                    if op_name in name:
                        need_ignore = True
                if need_ignore:
                    ignore_op_name_.append(name)
                else:
                    convert_op_name_.append(name)
            else:
                convert_op_name_.append(name)

    for name in convert_op_name_:
        name_str = name.split('.')
        ops = model
        if len(name_str) > 1:            
            for i in range(len(name_str)-1):
                if name_str[i].isdigit():
                    ops = ops[int(name_str[i])]
                else:
                    ops = getattr(ops, name_str[i])

        src_op = getattr(ops, name_str[-1])
        low_rank_op = LowRankConv2d(
                in_channels = src_op.in_channels,
                out_channels = src_op.out_channels,
                kernel_size = src_op.kernel_size[0],
                stride = src_op.stride,
                padding = src_op.padding,
                dilation = src_op.dilation,
                groups = src_op.groups,
                bias = src_op.bias is not None,
                device = src_op.weight.device,
                r = r,
                r_ratio = r_ratio,
                fix_sparse = args['fix_sparse'], 
                fix_low_rank = args['fix_low_rank'],
                tune_U = args['tune_U'], 
                tune_V = args['tune_V'],
                tune_V_S = args['tune_V_S'], 
                tune_U_S = args['tune_U_S'],
                tune_all = args['tune_all'],
                keep_noise = args['keep_noise'],
                reshape_consecutive = args['reshape_consecutive'],
                decompose_no_s = args['decompose_no_s'], 
                lora_mode = args['lora_mode'],
            )
        low_rank_op.weight.data = src_op.weight.data
        if src_op.bias is not None:
            low_rank_op.bias.data = src_op.bias.data
        setattr(ops, name_str[-1], low_rank_op)
        
        if mode != 'lora_mode' and not os.path.exists(path):
            print('reset weights ', name)
            if torch.cuda.is_available():
                _, _ = getattr(ops, name_str[-1]).cuda().decompose(compress_step, lambda_s)
            else:
                _, _ = getattr(ops, name_str[-1]).decompose(compress_step, lambda_s)
            if merge_parameter:
                getattr(ops, name_str[-1]).merge()
    if mode != 'lora_mode':
        if os.path.exists(path):
            model.load_state_dict(torch.load(path)['state_dict'])
        else:
            torch.save({'state_dict' : model.state_dict()}, path)
    print("--- convert results ---")
    print("convert conv op name : ")
    print(convert_op_name_)
    print("ignore conv op name : ")
    print(ignore_op_name_)
    print("fix conv op name : ")
    print(fix_op_name_)
    return model

def save_lora_state_dict(model, path):
    lora_state_dict = {}
    for k, v in model.state_dict().items():
        name_str = k.split('.')
        ops = model
        if len(name_str) > 0:            
            for i in range(len(name_str)):
                if name_str[i].isdigit():
                    ops = ops[int(name_str[i])]
                else:
                    ops = getattr(ops, name_str[i])
        if ops.requires_grad or name_str[-1] == 'running_mean' or name_str[-1] == 'running_var':
            lora_state_dict[k] = v

    torch.save(lora_state_dict, path)

def load_lora_state_dict(model, lora_path):
    src_keys = model.state_dict().keys()
    new_state_dict = {}
    state_dict = torch.load(lora_path)
    for k, v in state_dict.items():
        if k in src_keys:
            new_state_dict[k] = v
        else:
            if 'lora_A' in k:
                if k.replace('lora_A', 'sparse_weight') in state_dict.keys():
                    src_weight = model.state_dict()[k.replace('lora_A', 'weight')] = \
                        (state_dict[k.replace('lora_A', 'lora_B')] @ v).view(src_weight.shape) + \
                        state_dict[k.replace('lora_A', 'sparse_weight')] * \
                        (1 - state_dict[k.replace('lora_A', 'sparse_mask')])
                else:
                    src_weight = model.state_dict()[k.replace('lora_A', 'weight')]
                    new_state_dict[k.replace('lora_A', 'weight')] = \
                        src_weight + (state_dict[k.replace('lora_A', 'lora_B')] @ v).view(src_weight.shape)

    model.load_state_dict(new_state_dict, False)

if __name__ == '__main__':
    from torchvision.models.resnet import resnet18
    model = resnet18()

    model = model_convert_conv(model, mode='tune_U_S', path='weights.pth', ignore_op_name=['downsample'])

    for name, op in model.named_modules():
        print(name, type(op))
