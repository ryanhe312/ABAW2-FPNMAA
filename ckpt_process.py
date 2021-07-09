import torch
a = torch.load('fpn_multi.ckpt')
a['hyper_parameters']['data_type']='VA_Set' 
a['state_dict']['head.linear.bias'] = a['state_dict']["VA_head.linear.bias"]
a['state_dict']['head.linear.weight'] = a['state_dict']["VA_head.linear.weight"]
torch.save(a, 'multi_va.ckpt') 
a['hyper_parameters']['data_type']='EXPR_Set'
a['state_dict']['head.linear.bias'] = a['state_dict']["EXPR_head.linear.bias"]
a['state_dict']['head.linear.weight'] = a['state_dict']["EXPR_head.linear.weight"]
torch.save(a, 'multi_expr.ckpt')
a['hyper_parameters']['data_type']='AU_Set'
a['state_dict']['head.linear.bias'] = a['state_dict']["AU_head.linear.bias"]
a['state_dict']['head.linear.weight'] = a['state_dict']["AU_head.linear.weight"]
torch.save(a, 'multi_au.ckpt')
