# 指定master addr 、master port
export MASTER_ADDR="141.61.41.134" 
export MASTER_PORT="29500"
# 指定网卡
export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_SOCKET_IFNAME=enp189s0f0
# 拉起attn
python chat.py
# 拉起ffn
python ffn.py
