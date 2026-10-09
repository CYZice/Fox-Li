# 姓名：陈昀泽
# 学号：U202413823
# 班级：光实2401
from base import run_base
from plot_results import set_plot_style
from tilt import run_tilt

# 统一绘图风格
set_plot_style()

# 基础任务：没有扰动的一维、二维自再现模式
run_base()

# 进阶任务：单镜倾斜
run_tilt()
