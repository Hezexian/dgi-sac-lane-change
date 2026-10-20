"""
baseline SAC 的 encoder 预留位置。

当前 sac_base 仍然直接使用两层 MLP policy / critic，
所以这里先不放具体 encoder 实现。

后面如果要派生：
- sac_gcn
- sac_attention
- sac_gcn_ere

建议在各自算法目录下新增或扩展这一层，而不是再把编码逻辑散回根目录。
"""

__all__ = []
