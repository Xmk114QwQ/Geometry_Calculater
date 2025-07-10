import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, colorchooser
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import proj3d
import numpy as np
import time

class GeometryAnalyzer:
    def __init__(self):
        self.points = {}       # {点名: (x, y, z)} 存储数值坐标
        self.segments = {}     # {线段名: (起点, 终点, 颜色, 线型)} 存储线段信息
        self.vectors_to_display = []  # 需要显示的向量列表（包含起点、终点、颜色、标签）
        
        # 交互模式状态
        self.interaction_mode = False
        self.selected_points = []  # 存储用户选择的点
        self.circles_and_spheres = {}  # 存储圆和球的信息
        self.functions = {}  # {name: {'expr': str, 'var': str, 'range': tuple, 'is_3d': bool, 'color': str, 'linestyle': str}}

    def add_point(self, name, x, y, z=0):
        """添加点（自动转换符号表达式为数值）"""
        if name in self.points:
            return False, f"点 '{name}' 已存在"
        
        try:
            # 尝试将坐标转换为数值
            x_val = float(sp.sympify(x).evalf())
            y_val = float(sp.sympify(y).evalf())
            z_val = float(sp.sympify(z).evalf())
        except Exception as e:
            return False, f"坐标解析错误: {str(e)}"
        
        self.points[name] = (x_val, y_val, z_val)
        return True, f"成功添加点 '{name}'"

    def clear_temp_points(self):
        """清除所有以'temp_'开头的临时点"""
        temp_point_names = [name for name in self.points if name.startswith('temp_')]
        for name in temp_point_names:
            del self.points[name]
        
        # 清除向量显示列表中起点或终点为临时点的向量
        self.vectors_to_display = [vec for vec in self.vectors_to_display 
                                  if not (vec['start'].startswith('temp_') or vec['end'].startswith('temp_'))]

    def add_segment(self, start, end, color='#0000FF', linestyle='solid'):
        """添加线段（自动验证点存在性）"""
        if start not in self.points or end not in self.points:
            return False, "起点或终点不存在"
        seg_name = f"{start}_{end}"  # 修改线段命名规则，避免重复
        if seg_name in self.segments:
            return False, f"线段 '{seg_name}' 已存在"
        
        self.segments[seg_name] = (start, end, color, linestyle)
        return True, f"成功添加线段 '{seg_name}'"

    def get_vector(self, start_point, end_point):
        """获取从起点到终点的向量坐标"""
        if start_point not in self.points or end_point not in self.points:
            return None
        start = np.array(self.points[start_point])
        end = np.array(self.points[end_point])
        return end - start  # 向量=终点-起点

    # ---------------------------- 向量计算核心方法 ----------------------------
    def vector_add(self, vec1, vec2):
        """向量加法"""
        return np.add(vec1, vec2)

    def vector_subtract(self, vec1, vec2):
        """向量减法"""
        return np.subtract(vec1, vec2)

    def vector_dot(self, vec1, vec2):
        """向量点积"""
        return np.dot(vec1, vec2)

    def vector_cross(self, vec1, vec2):
        """向量叉积"""
        return np.cross(vec1, vec2)

    def vector_magnitude(self, vec):
        """向量模长"""
        return np.linalg.norm(vec)

    def vector_angle(self, vec1, vec2, degrees=True):
        """向量夹角（默认角度制）"""
        dot = self.vector_dot(vec1, vec2)
        mag1 = self.vector_magnitude(vec1)
        mag2 = self.vector_magnitude(vec2)
        if mag1 == 0 or mag2 == 0:
            return None  # 零向量无夹角
        cos_theta = dot / (mag1 * mag2)
        # 处理浮点误差导致的cos_theta超出[-1,1]范围
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_rad = np.arccos(cos_theta)
        return np.degrees(theta_rad) if degrees else theta_rad

    def analyze_relations(self):
        """分析所有线段的几何关系"""
        relations = {
            'perpendicular': [],
            'parallel': [],
            'length_ratio': [],
            'length_equal': [],
            'length_diff': []
        }

        seg_names = list(self.segments.keys())
        for i in range(len(seg_names)):
            seg1 = seg_names[i]
            s1_start, s1_end, _, _ = self.segments[seg1]
            v1 = self.get_vector(s1_start, s1_end)
            
            for j in range(i+1, len(seg_names)):
                seg2 = seg_names[j]
                s2_start, s2_end, _, _ = self.segments[seg2]
                v2 = self.get_vector(s2_start, s2_end)

                # 垂直判断（点积为0）
                if abs(self.vector_dot(v1, v2)) < 1e-6:
                    relations['perpendicular'].append((seg1, seg2))

                # 平行判断（叉积为0）
                if np.allclose(self.vector_cross(v1, v2), [0, 0, 0]):
                    relations['parallel'].append((seg1, seg2))

                # 长度计算
                len1 = self.vector_magnitude(v1)
                len2 = self.vector_magnitude(v2)

                # 长度比
                if len2 != 0:
                    ratio = round(len1/len2, 2)
                    relations['length_ratio'].append((seg1, seg2, ratio))

                # 长度相等
                if abs(len1 - len2) < 1e-6:
                    relations['length_equal'].append((seg1, seg2))

                # 长度差
                if len1 != len2:
                    diff = round(abs(len1 - len2), 2)
                    relations['length_diff'].append((seg1, seg2, diff))

        return relations

    def get_status(self):
        """获取当前状态信息，包括向量信息"""
        point_details = [f"{name}({x:.2f}, {y:.2f}, {z:.2f})" 
                        for name, (x, y, z) in self.points.items()]
        
        segment_details = []
        for seg_name, (start, end, color, linestyle) in self.segments.items():
            s_x, s_y, s_z = self.points[start]
            e_x, e_y, e_z = self.points[end]
            length = self.vector_magnitude(np.array([e_x - s_x, e_y - s_y, e_z - s_z]))
            segment_details.append(f"{seg_name}({start}-{end}, 长度={length:.2f}, 颜色={color}, 线型={linestyle})")
        
        # 添加向量信息
        vector_details = []
        for vec_info in self.vectors_to_display:
            start = vec_info['start']
            end = vec_info['end']
            label = vec_info.get('label', '向量')
            
            # 获取起点和终点坐标
            s_coord = self.points.get(start, (0,0,0))
            e_coord = self.points.get(end, (0,0,0))
            
            # 计算向量模长
            vec = np.array(e_coord) - np.array(s_coord)
            mag = self.vector_magnitude(vec)
            
            vector_details.append(f"{label}: {start}→{end}, 长度={mag:.2f}, 颜色={vec_info['color']}")
        
        # 获取计算点信息
        calculation_details = [
            f"{name}({x:.2f}, {y:.2f}, {z:.2f})" 
            for name, (x, y, z) in self.points.items()
            if name.startswith('result_') or name.startswith('temp_')
        ]
        
        return {
            'points_count': len(self.points),
            'segments_count': len(self.segments),
            'vectors_count': len(self.vectors_to_display),
            'calculation_count': len(calculation_details),
            'point_details': point_details,
            'segment_details': segment_details,
            'vector_details': vector_details,
            'calculation_details': calculation_details,
            'circles_count': len(self.circles_and_spheres),
            'circle_details': [f"{name}(中心={info['center']}, 半径={info['radius']:.2f})" 
                  for name, info in self.circles_and_spheres.items()],
            'function_count': len(self.functions),
            'function_details': [f"{name}: {info['expr']} ({'3D' if info['is_3d'] else '2D'}, 范围={info['range']})"
                  for name, info in self.functions.items()],
        }
    
    def delete_vector(self, index):
        """删除指定索引的向量"""
        if 0 <= index < len(self.vectors_to_display):
            del self.vectors_to_display[index]
            return True
        return False
    
    def clear_all_vectors(self):
        """清除所有向量显示"""
        self.vectors_to_display = []
    
    def clear_all_calculations(self):
        """清除所有计算相关点和向量"""
        self.clear_all_vectors()
        
        # 清除临时点和结果点
        temp_point_names = [
            name for name in self.points 
            if name.startswith('temp_') or name.startswith('result_')
        ]
        for name in temp_point_names:
            del self.points[name]
    
    def create_perpendicular(self, point_name, base_segment_name, result_name_prefix="perp"):
        """过某点做某线段的垂直线"""
        if point_name not in self.points or base_segment_name not in self.segments:
            return False, "点或线段不存在"
        
        # 获取线段信息
        start, end, _, _ = self.segments[base_segment_name]
        x0, y0, z0 = self.points[point_name]
        x1, y1, _ = self.points[start]
        x2, y2, _ = self.points[end]
        
        # 计算线段方向向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 计算垂直线方向向量 (旋转90度)
        perp_dx = -dy
        perp_dy = dx
        
        # 创建垂直线终点
        end_x = x0 + perp_dx
        end_y = y0 + perp_dy
        
        # 生成唯一名称
        timestamp = int(time.time() * 1000)
        end_point_name = f"{result_name_prefix}_end_{timestamp}"
        
        # 添加点和线段
        self.add_point(end_point_name, end_x, end_y, z0)
        seg_name = f"{point_name}_{end_point_name}"
        self.add_segment(point_name, end_point_name, '#FF00FF', 'dashed')
        
        return True, f"成功创建垂直线 {seg_name}"

    def create_parallel(self, point_name, base_segment_name, result_name_prefix="parallel"):
        """过某点做某线段的平行线"""
        if point_name not in self.points or base_segment_name not in self.segments:
            return False, "点或线段不存在"
        
        # 获取线段信息
        start, end, _, _ = self.segments[base_segment_name]
        x0, y0, z0 = self.points[point_name]
        x1, y1, _ = self.points[start]
        x2, y2, _ = self.points[end]
        
        # 计算线段方向向量
        dx = x2 - x1
        dy = y2 - y1
        
        # 创建平行线终点
        end_x = x0 + dx
        end_y = y0 + dy
        
        # 生成唯一名称
        timestamp = int(time.time() * 1000)
        end_point_name = f"{result_name_prefix}_end_{timestamp}"
        
        # 添加点和线段
        self.add_point(end_point_name, end_x, end_y, z0)
        seg_name = f"{point_name}_{end_point_name}"
        self.add_segment(point_name, end_point_name, '#00AAFF', 'dashed')
        
        return True, f"成功创建平行线 {seg_name}"

    def create_midpoint(self, segment_name, result_name_prefix="mid"):
        """创建某线段的中点"""
        if segment_name not in self.segments:
            return False, "线段不存在"
        
        start, end, _, _ = self.segments[segment_name]
        x1, y1, z1 = self.points[start]
        x2, y2, z2 = self.points[end]
        
        # 计算中点坐标
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        mid_z = (z1 + z2) / 2
        
        # 生成唯一名称
        timestamp = int(time.time() * 1000)
        mid_point_name = f"{result_name_prefix}_{segment_name}_{timestamp}"
        
        # 添加点
        self.add_point(mid_point_name, mid_x, mid_y, mid_z)
        
        return True, f"成功创建中点 {mid_point_name}"

    def create_circle_or_sphere(self, center_point, radius_segment, result_name_prefix="circle"):
        """以某点为圆心，某线段长度为半径创建圆(2D)或球(3D)"""
        if center_point not in self.points or radius_segment not in self.segments:
            return False, "点或线段不存在"
        
        # 获取圆心坐标
        cx, cy, cz = self.points[center_point]
        
        # 计算半径长度
        start, end, _, _ = self.segments[radius_segment]
        x1, y1, z1 = self.points[start]
        x2, y2, z2 = self.points[end]
        radius = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        
        # 生成唯一名称
        timestamp = int(time.time() * 1000)
        circle_name = f"{result_name_prefix}_{timestamp}"
        
        # 添加圆/球信息
        self.circles_and_spheres[circle_name] = {
            'center': center_point,
            'radius': radius,
            'color': '#FFA500',
            'linestyle': 'dashed'
        }
        
        return True, f"成功创建圆/球 {circle_name} (半径={radius:.2f})"
    
    def add_function(self, name, expr, var, range_vals, color='#0000FF', linestyle='solid', is_3d=False):
        if name in self.functions:
            return False, f"名称 '{name}' 已存在"
        try:
            if is_3d:
                x, y = sp.symbols(var.split(','))
                sp.sympify(expr)
            else:
                x = sp.symbols(var)
                sp.sympify(expr)
        except Exception as e:
            return False, f"函数解析错误: {str(e)}"
        self.functions[name] = {
            'expr': expr,
            'var': var,
            'range': range_vals,
            'color': color,
            'linestyle': linestyle,
            'is_3d': is_3d
        }
        return True, f"函数 '{name}' 已添加"

class GeometryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能几何分析器（含向量计算与可视化）")
        self.root.geometry("1200x800")
        self.root.configure(bg="#FFFFFF")  # 纯白色背景
        # 初始化选中的点列表
        self.selected_points = []  # 存储用户选择的点
        
        # 配置更现代的主题（白色版）
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#FFFFFF")
        style.configure("TFrame", background="#FFFFFF")
        style.configure("TButton", background="#4a86e8", foreground="white")
        style.map("TButton", background=[('active', '#3a76d8')])
        style.configure("TLabel", background="#FFFFFF", foreground="black")
        style.configure("TLabelframe", background="#FFFFFF")
        style.configure("TLabelframe.Label", background="#FFFFFF", foreground="black")
        style.configure("TNotebook", background="#FFFFFF")
        style.configure("TNotebook.Tab", background="#f0f0f0", padding=[10, 5], foreground="black")
        style.map("TNotebook.Tab", background=[("selected", "#FFFFFF")])
        style.configure("TCombobox", fieldbackground="white")
        style.configure("Vertical.TScrollbar", background="#e0e0e0")
        
        # 初始化分析器
        self.analyzer = GeometryAnalyzer()
        
        # 创建主框架
        self.main_frame = ttk.Frame(self.root, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ==================== 顶部工具栏 ====================
        top_toolbar = ttk.Frame(self.main_frame)
        top_toolbar.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        # 使用 Grid 布局来控制按钮分布
        for i in range(4):  # 四列
            top_toolbar.grid_columnconfigure(i, weight=1)

        # 添加按钮
        ttk.Button(top_toolbar, text="切换视图", command=self.toggle_3d_view).grid(
            row=0, column=0, sticky="ew", padx=2)

        ttk.Button(top_toolbar, text="加载示例", command=self._add_sample_data).grid(
            row=0, column=1, sticky="ew", padx=2)

        ttk.Button(top_toolbar, text="刷新页面", command=self.refresh_page).grid(
            row=0, column=2, sticky="ew", padx=2)

        ttk.Button(top_toolbar, text="交互模式", command=self.toggle_interaction_mode).grid(
            row=0, column=3, sticky="ew", padx=2)

        # 创建主分割窗口 (PanedWindow) - 可拖动的分隔条
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板框架 (可调整大小)
        self.control_container = tk.Frame(self.paned_window)
        self.paned_window.add(self.control_container, weight=1)  # 初始权重设为1 (约占总宽的25%)
        
        # 设置分割条样式
        style.configure("Sash", background="#d0d0d0", gripcount=5)
        
        # 右侧绘图区域框架
        self.plot_container = ttk.Frame(self.paned_window)
        self.paned_window.add(self.plot_container, weight=3)  # 初始权重设为3 (约占总宽的75%)
        
        # 创建画布和滚动条 (左侧控制面板内)
        self.canvas = tk.Canvas(self.control_container, bg="#FFFFFF", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.control_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # 配置画布滚动
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 绑定鼠标滚轮事件
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 绑定画布大小变化事件
        self.canvas.bind("<Configure>", self._on_canvas_resize)
        
        # 布局画布和滚动条
        self.canvas.pack(side="left", fill="both", expand=True, padx=(0, 2))
        self.scrollbar.pack(side="right", fill="y", padx=(0, 2))
        
        # 控制面板框架
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="控制面板", padding=10)
        self.control_frame.pack(fill=tk.X, expand=True)
        
        # 添加3D切换按钮
        self.btn_switch_3d = ttk.Button(self.control_frame, text="切换3D视图", 
                                      command=self.toggle_3d_view)
        self.btn_switch_3d.pack(fill=tk.X, pady=5)
        
        # 初始化线段样式默认值
        self.segment_color = '#0000FF'  # 默认蓝色
        self.segment_linestyle = 'solid'  # 默认实线
        
        # 创建功能选项卡
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 几何操作选项卡
        self.geo_ops_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.geo_ops_tab, text="几何操作")
        self._setup_geo_ops_tab()
        
        # 向量计算选项卡（新增）
        self.vector_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.vector_tab, text="向量计算")
        self._setup_vector_tab()
        
        # 分析选项卡
        self.analysis_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.analysis_tab, text="分析结果")
        self._setup_analysis_tab()
        
        # 状态选项卡
        self.status_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.status_tab, text="状态信息")
        self._setup_status_tab()
        
        # 右侧绘图区框架
        self.plot_frame = ttk.LabelFrame(self.plot_container, text="几何图形", padding=10)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建绘图区域和工具栏
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.canvas_plot = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加Matplotlib导航工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.plot_frame)
        self.toolbar.update()

        # 添加交互模式切换按钮到控制面板
        self.btn_interaction = ttk.Button(self.control_frame, text="交互模式: 关闭", 
                                        command=self.toggle_interaction_mode)
        self.btn_interaction.pack(fill=tk.X, pady=5)
        
        # 交互模式状态
        self.interaction_mode = False
        self.selected_points = []  # 存储用户选择的点
        
        # 绑定图形点击事件
        self.canvas_plot.mpl_connect('button_press_event', self.on_plot_click)
        
        # 初始化2D绘图
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("white")  # 白色背景
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("2D几何视图", color="black")
        self.ax.tick_params(colors='black')
        self.ax.set_aspect('equal')  # 确保2D视图等比例
        
        # 当前视图模式
        self.current_view = '2d'
        
        # 初始化示例数据
        self._add_sample_data()
        
        # 绑定关闭窗口事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_left_panel(self):
        """设置左侧控制面板（带滚动条）"""
        # 创建画布和滚动条
        self.canvas = tk.Canvas(self.control_container, bg="#FFFFFF", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.control_container, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        # 配置画布滚动
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        # 绑定鼠标滚轮事件
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        # 布局画布和滚动条
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 控制面板框架
        self.control_frame = ttk.LabelFrame(self.scrollable_frame, text="控制面板", padding=10)
        self.control_frame.pack(fill=tk.X, expand=True, pady=5)
        
        # 添加3D切换按钮
        self.btn_switch_3d = ttk.Button(self.control_frame, text="切换3D视图", 
                                    command=self.toggle_3d_view)
        self.btn_switch_3d.pack(fill=tk.X, pady=5)
        
        # 创建功能选项卡
        self.notebook = ttk.Notebook(self.control_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 几何操作选项卡
        self.geo_ops_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.geo_ops_tab, text="几何操作")
        self._setup_geo_ops_tab()
        
        # 向量计算选项卡
        self.vector_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.vector_tab, text="向量计算")
        self._setup_vector_tab()
        
        # 分析选项卡
        self.analysis_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.analysis_tab, text="分析结果")
        self._setup_analysis_tab()
        
        # 状态选项卡
        self.status_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.status_tab, text="状态信息")
        self._setup_status_tab()

    def _setup_right_panel(self):
        """设置右侧绘图区域"""
        # 右侧绘图区框架
        self.plot_frame = ttk.LabelFrame(self.plot_container, text="几何图形", padding=10)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建绘图区域和工具栏
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor="white")
        self.canvas_plot = FigureCanvasTkAgg(self.fig, self.plot_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 添加Matplotlib导航工具栏
        self.toolbar = NavigationToolbar2Tk(self.canvas_plot, self.plot_frame)
        self.toolbar.update()
        
        # 初始化2D绘图
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("white")  # 白色背景
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("2D几何视图", color="black")
        self.ax.tick_params(colors='black')
        self.ax.set_aspect('equal')  # 确保2D视图等比例
        
        # 当前视图模式
        self.current_view = '2d'


    def _on_canvas_resize(self, event):
        """当画布大小变化时调整滚动框架宽度"""
        # 设置滚动框架宽度等于画布可见宽度
        self.canvas.itemconfig("all", width=event.width)
        self.scrollable_frame.config(width=event.width)

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件"""
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _setup_geo_ops_tab(self):
        """设置几何操作选项卡（优化版）"""
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", background="#4CAF50")
        style.configure("TCombobox", padding=5)

        # ==================== 添加点区域 ====================
        point_frame = ttk.LabelFrame(self.geo_ops_tab, text="添加新点", padding=10)
        point_frame.pack(fill=tk.X, pady=5, expand=True)
        point_frame.columnconfigure(1, weight=1)

        fields = [
            ("点名称:", "point_name", tk.StringVar()),
            ("X坐标:", "point_x", tk.StringVar()),
            ("Y坐标:", "point_y", tk.StringVar()),
            ("Z坐标:", "point_z", tk.StringVar(value="0")),
        ]

        for i, (label_text, attr_name, var) in enumerate(fields):
            setattr(self, attr_name, var)
            ttk.Label(point_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, pady=5)
            entry = ttk.Entry(point_frame, textvariable=var)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(point_frame, text="添加点", command=self.add_point).grid(
            row=len(fields), column=0, columnspan=2, sticky="ew", pady=10, padx=5
        )

        # ==================== 添加线段区域 ====================
        segment_frame = ttk.LabelFrame(self.geo_ops_tab, text="添加线段", padding=10)
        segment_frame.pack(fill=tk.X, pady=5, expand=True)
        segment_frame.columnconfigure(1, weight=1)

        self.start_point = tk.StringVar()
        self.end_point = tk.StringVar()

        ttk.Label(segment_frame, text="起点:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.start_combo = ttk.Combobox(segment_frame, textvariable=self.start_point, state="readonly")
        self.start_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(segment_frame, text="终点:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.end_combo = ttk.Combobox(segment_frame, textvariable=self.end_point, state="readonly")
        self.end_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        # 颜色选择
        ttk.Label(segment_frame, text="颜色:").grid(row=2, column=0, sticky=tk.W, pady=5)
        color_frame = ttk.Frame(segment_frame)
        color_frame.grid(row=2, column=1, sticky="ew", pady=5)
        self.color_preview = tk.Canvas(color_frame, width=20, height=20, bg="black")
        self.color_preview.pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(color_frame, text="选择颜色", command=self.choose_segment_color).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )

        # 线型选择
        ttk.Label(segment_frame, text="线型:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.linestyle_var = tk.StringVar(value='solid')
        ttk.Combobox(
            segment_frame,
            textvariable=self.linestyle_var,
            state="readonly",
            values=['solid', 'dashed', 'dotted', 'dashdot']
        ).grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(segment_frame, text="添加线段", command=self.add_segment).grid(
            row=4, column=0, columnspan=2, sticky="ew", pady=10, padx=5
        )

        # ==================== 删除操作区域 ====================
        delete_frame = ttk.LabelFrame(self.geo_ops_tab, text="删除操作", padding=10)
        delete_frame.pack(fill=tk.X, pady=5, expand=True)
        delete_frame.columnconfigure(1, weight=1)

        self.delete_type = tk.StringVar(value='点')
        self.delete_object = tk.StringVar()

        ttk.Label(delete_frame, text="删除类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Combobox(delete_frame, textvariable=self.delete_type, state="readonly",
                    values=['点', '线段']).grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(delete_frame, text="选择对象:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.delete_combo = ttk.Combobox(delete_frame, textvariable=self.delete_object, state="readonly")
        self.delete_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(delete_frame, text="删除", command=self.delete_object_action).grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=10, padx=5
        )

        # ==================== 向量管理区域 ====================
        vector_frame = ttk.LabelFrame(self.geo_ops_tab, text="向量管理", padding=10)
        vector_frame.pack(fill=tk.X, pady=5, expand=True)
        vector_frame.columnconfigure(1, weight=1)

        ttk.Label(vector_frame, text="删除向量:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.vector_delete_combo = ttk.Combobox(vector_frame, state="readonly")
        self.vector_delete_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(vector_frame, text="删除选定向量", command=self.delete_selected_vector).grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Button(vector_frame, text="清除所有向量", command=self.clear_all_vectors).grid(
            row=0, column=3, padx=5, pady=5
        )

        # ==================== 计算结果管理区域 ====================
        calc_frame = ttk.LabelFrame(self.geo_ops_tab, text="计算结果管理", padding=10)
        calc_frame.pack(fill=tk.X, pady=5, expand=True)
        calc_frame.columnconfigure(1, weight=1)

        ttk.Label(calc_frame, text="删除计算结果:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.calculation_delete_combo = ttk.Combobox(calc_frame, state="readonly")
        self.calculation_delete_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Button(calc_frame, text="删除选定结果", command=self.delete_selected_calculation).grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Button(calc_frame, text="清除所有计算结果", command=self.clear_all_calculations).grid(
            row=0, column=3, padx=5, pady=5
        )

        # ==================== 几何作图区域 ====================
        construction_frame = ttk.LabelFrame(self.geo_ops_tab, text="几何作图", padding=10)
        construction_frame.pack(fill=tk.X, pady=5, expand=True)
        for i in range(6):
            construction_frame.columnconfigure(i, weight=1)

        # 垂直线
        ttk.Label(construction_frame, text="过").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.perp_point = tk.StringVar()
        self.perp_point_combo = ttk.Combobox(construction_frame, textvariable=self.perp_point, state="readonly")
        self.perp_point_combo.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="作").grid(row=0, column=2, sticky=tk.W, pady=2)
        self.perp_segment = tk.StringVar()
        self.perp_segment_combo = ttk.Combobox(construction_frame, textvariable=self.perp_segment, state="readonly")
        self.perp_segment_combo.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="的垂线").grid(row=0, column=4, sticky=tk.W, pady=2)
        ttk.Button(construction_frame, text="创建", command=self.draw_perpendicular).grid(
            row=0, column=5, padx=5, pady=2, sticky="ew"
        )

        # 平行线
        ttk.Label(construction_frame, text="过").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.parallel_point = tk.StringVar()
        self.parallel_point_combo = ttk.Combobox(construction_frame, textvariable=self.parallel_point, state="readonly")
        self.parallel_point_combo.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="作").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.parallel_segment = tk.StringVar()
        self.parallel_segment_combo = ttk.Combobox(construction_frame, textvariable=self.parallel_segment, state="readonly")
        self.parallel_segment_combo.grid(row=1, column=3, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="的平行线").grid(row=1, column=4, sticky=tk.W, pady=2)
        ttk.Button(construction_frame, text="创建", command=self.draw_parallel).grid(
            row=1, column=5, padx=5, pady=2, sticky="ew"
        )

        # 中点
        ttk.Label(construction_frame, text="作").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.mid_segment = tk.StringVar()
        self.mid_segment_combo = ttk.Combobox(construction_frame, textvariable=self.mid_segment, state="readonly")
        self.mid_segment_combo.grid(row=2, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="的中点").grid(row=2, column=2, sticky=tk.W, pady=2)
        ttk.Button(construction_frame, text="创建", command=self.draw_midpoint).grid(
            row=2, column=5, padx=5, pady=2, sticky="e"
        )

        # 圆/球
        ttk.Label(construction_frame, text="以").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.circle_center = tk.StringVar()
        self.circle_center_combo = ttk.Combobox(construction_frame, textvariable=self.circle_center, state="readonly")
        self.circle_center_combo.grid(row=3, column=1, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="为圆心").grid(row=3, column=2, sticky=tk.W, pady=2)
        self.circle_radius = tk.StringVar()
        self.circle_radius_combo = ttk.Combobox(construction_frame, textvariable=self.circle_radius, state="readonly")
        self.circle_radius_combo.grid(row=3, column=3, padx=5, pady=2, sticky="ew")
        ttk.Label(construction_frame, text="为半径作圆/球").grid(row=3, column=4, sticky=tk.W, pady=2)
        ttk.Button(construction_frame, text="创建", command=self.draw_circle).grid(
            row=3, column=5, padx=5, pady=2, sticky="ew"
        )

        # 新增函数输入区域
        function_frame = ttk.LabelFrame(self.geo_ops_tab, text="函数输入", padding=10)
        function_frame.pack(fill=tk.X, pady=5, expand=True)

        ttk.Label(function_frame, text="函数名称:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.function_name = tk.StringVar()
        ttk.Entry(function_frame, textvariable=self.function_name).grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(function_frame, text="函数表达式:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.function_expr = tk.StringVar()
        ttk.Entry(function_frame, textvariable=self.function_expr).grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(function_frame, text="变量:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.function_var = tk.StringVar(value="x")
        ttk.Entry(function_frame, textvariable=self.function_var).grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(function_frame, text="范围:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.function_range_min = tk.StringVar(value="0")
        self.function_range_max = tk.StringVar(value="10")
        ttk.Entry(function_frame, textvariable=self.function_range_min, width=5).grid(row=3, column=1, sticky="w", padx=5, pady=2)
        ttk.Label(function_frame, text="到").grid(row=3, column=1, sticky=tk.E, padx=5, pady=2)
        ttk.Entry(function_frame, textvariable=self.function_range_max, width=5).grid(row=3, column=1, sticky="e", padx=5, pady=2)

        ttk.Label(function_frame, text="类型:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.function_type = tk.StringVar(value="2d")
        ttk.Radiobutton(function_frame, text="2D", variable=self.function_type, value="2d").grid(row=4, column=1, sticky="w", padx=5, pady=2)
        ttk.Radiobutton(function_frame, text="3D", variable=self.function_type, value="3d").grid(row=4, column=1, sticky="e", padx=5, pady=2)

        ttk.Button(function_frame, text="添加函数", command=self.add_function).grid(
            row=5, column=0, columnspan=2, sticky="ew", pady=10, padx=5
        )

        # ==================== 删除函数区域 ====================
        delete_function_frame = ttk.LabelFrame(self.geo_ops_tab, text="删除函数", padding=10)
        delete_function_frame.pack(fill=tk.X, pady=5, expand=True)
        delete_function_frame.columnconfigure(1, weight=1)

        ttk.Label(delete_function_frame, text="函数名称:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.delete_function_name = tk.StringVar()
        self.delete_function_combo = ttk.Combobox(delete_function_frame,
                                                textvariable=self.delete_function_name,
                                                state="readonly")
        self.delete_function_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ttk.Button(delete_function_frame, text="删除函数", command=self.delete_function).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=10, padx=5
        )

        # 初始化下拉框内容
        self._update_delete_function_combo()

        # ==================== 绑定事件 ====================
        self.point_name.trace_add("write", lambda *args: self._update_combo_boxes())
        self.delete_type.trace_add("write", lambda *args: self._update_delete_combo())

    def _setup_vector_tab(self):
        """设置向量计算选项卡（新增核心功能，优化版）"""
        vector_frame = ttk.LabelFrame(self.vector_tab, text="向量输入", padding=10)
        vector_frame.pack(fill=tk.X, pady=5, expand=True)

        # 设置网格列权重
        for i in range(4):
            vector_frame.columnconfigure(i, weight=1 if i % 2 == 1 else 0)

        # 向量1输入
        ttk.Label(vector_frame, text="向量1 (起点→终点):").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(vector_frame, text="起点:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.vec1_start = tk.StringVar()
        self.vec1_start_combo = ttk.Combobox(vector_frame, textvariable=self.vec1_start, state="readonly")
        self.vec1_start_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        ttk.Label(vector_frame, text="终点:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.vec1_end = tk.StringVar()
        self.vec1_end_combo = ttk.Combobox(vector_frame, textvariable=self.vec1_end, state="readonly")
        self.vec1_end_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=2)

        # 向量2输入
        ttk.Label(vector_frame, text="向量2 (起点→终点):").grid(row=0, column=2, columnspan=2, sticky=tk.W, pady=5)
        ttk.Label(vector_frame, text="起点:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.vec2_start = tk.StringVar()
        self.vec2_start_combo = ttk.Combobox(vector_frame, textvariable=self.vec2_start, state="readonly")
        self.vec2_start_combo.grid(row=1, column=3, sticky="ew", padx=5, pady=2)

        ttk.Label(vector_frame, text="终点:").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.vec2_end = tk.StringVar()
        self.vec2_end_combo = ttk.Combobox(vector_frame, textvariable=self.vec2_end, state="readonly")
        self.vec2_end_combo.grid(row=2, column=3, sticky="ew", padx=5, pady=2)

        # 直接输入坐标
        ttk.Label(vector_frame, text="(可选) 直接输入向量坐标:").grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(10, 2))
        self.vec1_input = tk.StringVar(value="0,0,0")
        ttk.Entry(vector_frame, textvariable=self.vec1_input).grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        self.vec2_input = tk.StringVar(value="0,0,0")
        ttk.Entry(vector_frame, textvariable=self.vec2_input).grid(row=4, column=2, columnspan=2, sticky="ew", padx=5, pady=2)

        # 计算类型选择
        ttk.Label(vector_frame, text="计算类型:").grid(row=5, column=0, sticky=tk.W, pady=5)
        self.calc_type = tk.StringVar(value="点积")
        calc_combo = ttk.Combobox(vector_frame, textvariable=self.calc_type, state="readonly",
                                values=["加法", "减法", "点积", "叉积", "模长(向量1)", "模长(向量2)", "夹角"])
        calc_combo.grid(row=5, column=1, sticky="ew", padx=5, pady=5)

        # 计算按钮
        ttk.Button(vector_frame, text="执行计算", command=self.calculate_vector).grid(
            row=5, column=2, columnspan=2, sticky="ew", padx=5, pady=5)

        # 结果显示区域
        result_frame = ttk.LabelFrame(self.vector_tab, text="计算结果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.vector_result = scrolledtext.ScrolledText(result_frame,
                                                    bg="white",
                                                    fg="black",
                                                    insertbackground="black",
                                                    font=("Consolas", 10),
                                                    height=10)
        self.vector_result.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.vector_result.insert(tk.END, "选择向量并点击'执行计算'查看结果...\n")
        self.vector_result.config(state=tk.DISABLED)

        # 绑定点选择事件更新向量输入
        points = list(self.analyzer.points.keys())
        self.vec1_start_combo['values'] = points
        self.vec1_end_combo['values'] = points
        self.vec2_start_combo['values'] = points
        self.vec2_end_combo['values'] = points

        if points:
            self.vec1_start_combo.current(0)
            self.vec1_end_combo.current(1 if len(points) > 1 else 0)
            self.vec2_start_combo.current(0)
            self.vec2_end_combo.current(1 if len(points) > 1 else 0)

    def _setup_analysis_tab(self):
        """设置分析选项卡（优化版）"""
        result_frame = ttk.Frame(self.analysis_tab)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 分析按钮
        analyze_btn = ttk.Button(result_frame, text="运行几何分析", command=self.analyze_geometry)
        analyze_btn.pack(fill=tk.X, pady=(0, 15))

        # 结果标题
        ttk.Label(result_frame, text="几何关系分析结果:").pack(anchor=tk.W, pady=(0, 5))

        # 结果文本框
        self.result_text = scrolledtext.ScrolledText(result_frame,
                                                    bg="white",
                                                    fg="black",
                                                    insertbackground="black",
                                                    font=("Consolas", 9),
                                                    height=15)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.result_text.insert(tk.END, "点击上方按钮分析几何关系...\n")
        self.result_text.insert(tk.END, "结果将显示在此区域\n")
        self.result_text.config(state=tk.DISABLED)

    def _setup_status_tab(self):
        """设置状态选项卡（优化版）"""
        status_frame = ttk.Frame(self.status_tab)
        status_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # 状态文本框
        self.status_text = scrolledtext.ScrolledText(status_frame,
                                                    bg="white",
                                                    fg="black",
                                                    insertbackground="black",
                                                    font=("Consolas", 9),
                                                    height=15)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_text.insert(tk.END, "状态信息将显示在此\n")
        self.status_text.config(state=tk.DISABLED)

    def choose_segment_color(self):
        """打开颜色选择对话框"""
        color = colorchooser.askcolor(title="选择线段颜色", initialcolor=self.segment_color)
        if color[1]:  # 用户选择了颜色
            self.segment_color = color[1]
            self.color_preview.config(bg=self.segment_color)

    def _update_combo_boxes(self):
        """更新所有下拉框"""
        points = list(self.analyzer.points.keys())
        segments = list(self.analyzer.segments.keys())
        
        self.start_combo['values'] = points
        self.end_combo['values'] = points
        
        if points:
            if not self.start_combo.get():
                self.start_combo.current(0)
            if not self.end_combo.get():
                self.end_combo.current(0)
        
        # 更新删除下拉框
        self._update_delete_combo()
        # 更新向量输入下拉框
        self.vec1_start_combo['values'] = points
        self.vec1_end_combo['values'] = points
        self.vec2_start_combo['values'] = points
        self.vec2_end_combo['values'] = points
        if points and len(points)>=2:
            self.vec1_start_combo.current(0)
            self.vec1_end_combo.current(min(1, len(points)-1))
            self.vec2_start_combo.current(0)
            self.vec2_end_combo.current(min(1, len(points)-1))
        
        # 更新向量删除下拉框
        vector_options = []
        for vec_info in self.analyzer.vectors_to_display:
            start = vec_info['start']
            end = vec_info['end']
            label = vec_info.get('label', '向量')
            vector_options.append(f"{label} ({start}→{end})")
        
        self.vector_delete_combo['values'] = vector_options
        if vector_options:
            self.vector_delete_combo.current(0)
        
        # 更新计算结果删除下拉框
        calculation_options = []
        for name in self.analyzer.points:
            if name.startswith('result_'):
                x, y, z = self.analyzer.points[name]
                calculation_options.append(f"{name}({x:.2f}, {y:.2f}, {z:.2f})")
        
        self.calculation_delete_combo['values'] = calculation_options
        if calculation_options:
            self.calculation_delete_combo.current(0)

        # 更新几何作图下拉框
        points = list(self.analyzer.points.keys())
        segments = list(self.analyzer.segments.keys())
        
        # 垂直线
        self.perp_point_combo['values'] = points
        self.perp_segment_combo['values'] = segments
        
        # 平行线
        self.parallel_point_combo['values'] = points
        self.parallel_segment_combo['values'] = segments
        
        # 中点
        self.mid_segment_combo['values'] = segments
        
        # 圆/球
        self.circle_center_combo['values'] = points
        self.circle_radius_combo['values'] = segments

    def _update_delete_combo(self):
        """更新删除下拉框内容"""
        delete_type = self.delete_type.get()
        
        if delete_type == "点":
            values = list(self.analyzer.points.keys())
        else:  # 线段
            values = list(self.analyzer.segments.keys())
        
        self.delete_combo['values'] = values
        
        if values:
            self.delete_combo.current(0)
        else:
            self.delete_object.set("")

    def _update_status(self):
        status = self.analyzer.get_status()
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)

        # 更新选项卡标题
        self.notebook.tab(3, text=f"状态信息 ({status['points_count']}点 {status['segments_count']}线段 "
                                f"{status['vectors_count']}向量 {status['calculation_count']}计算结果 "
                                f"{status['function_count']}函数)")

        # 显示状态信息
        self.status_text.insert(tk.END, f"点数量: {status['points_count']} 线段数量: {status['segments_count']} "
                                        f"向量数量: {status['vectors_count']} 计算结果数量: {status['calculation_count']} "
                                        f"函数数量: {status['function_count']}\n")

        if status['point_details']:
            self.status_text.insert(tk.END, "\n点坐标列表:\n")
            for detail in status['point_details']:
                self.status_text.insert(tk.END, f" • {detail}\n")

        if status['segment_details']:
            self.status_text.insert(tk.END, "\n线段信息:\n")
            for detail in status['segment_details']:
                self.status_text.insert(tk.END, f" • {detail}\n")

        if status['vector_details']:
            self.status_text.insert(tk.END, "\n向量信息:\n")
            for detail in status['vector_details']:
                self.status_text.insert(tk.END, f" • {detail}\n")

        if status['circle_details']:
            self.status_text.insert(tk.END, "\n圆/球信息:\n")
            for detail in status['circle_details']:
                self.status_text.insert(tk.END, f" • {detail}\n")

        if status['function_details']:
            self.status_text.insert(tk.END, "\n函数信息:\n")
            for detail in status['function_details']:
                self.status_text.insert(tk.END, f" • {detail}\n")

        self.status_text.config(state=tk.DISABLED)
    
    def _update_delete_function_combo(self):
        """更新删除函数下拉框"""
        function_names = list(self.analyzer.functions.keys())
        self.delete_function_combo['values'] = function_names
        if function_names:
            self.delete_function_combo.current(0)
        else:
            self.delete_function_name.set("")

    def _add_sample_data(self):
        """添加示例数据（修正后）"""
        # 清空旧数据
        self.analyzer.points.clear()
        self.analyzer.segments.clear()
        self.analyzer.vectors_to_display.clear()  # 清空向量显示列表
        
        # 添加基础点（包含Z轴方向）
        self.analyzer.add_point("O", 0, 0, 0)    # 原点
        self.analyzer.add_point("A", 3, 0, 0)     # X轴方向
        self.analyzer.add_point("B", 0, 3, 0)     # Y轴方向
        self.analyzer.add_point("C", 0, 0, 3)     # Z轴方向
        self.analyzer.add_point("D", 2, 2, 2)     # 空间点
        self.analyzer.add_point("E", 1, 2, 3)     # 额外点
        
        # 添加线段（使用正确的点命名）
        self.analyzer.add_segment("O", "A", '#FF0000', 'solid')   # 红色实线
        self.analyzer.add_segment("O", "B", '#00FF00', 'solid')   # 绿色实线
        self.analyzer.add_segment("O", "C", '#0000FF', 'solid')   # 蓝色实线
        self.analyzer.add_segment("O", "D", '#FF00FF', 'dashed') # 紫色虚线
        self.analyzer.add_segment("A", "B", '#FFA500', 'dotted')  # 橙色点线
        self.analyzer.add_segment("A", "C", '#800080', 'dashdot') # 紫色点划线
        
        # 更新界面
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()
        messagebox.showinfo("成功", "示例数据已加载")

    def add_point(self):
        """添加点操作（修正后）"""
        name = self.point_name.get().strip()
        x = self.point_x.get().strip()
        y = self.point_y.get().strip()
        z = self.point_z.get().strip()
        
        if not name:
            messagebox.showerror("错误", "点名称不能为空")
            return
            
        if not x or not y:
            messagebox.showerror("错误", "X坐标和Y坐标不能为空")
            return
        
        # 尝试计算坐标值
        try:
            x_val = float(sp.sympify(x).evalf())
            y_val = float(sp.sympify(y).evalf())
            z_val = float(sp.sympify(z).evalf()) if z else 0.0
        except Exception as e:
            messagebox.showerror("错误", f"坐标格式无效: {str(e)}")
            return
        
        success, msg = self.analyzer.add_point(name, x_val, y_val, z_val)
        if not success:
            messagebox.showerror("错误", msg)
        else:
            messagebox.showinfo("成功", msg)
        
        # 清空输入
        self.point_name.set("")
        self.point_x.set("")
        self.point_y.set("")
        self.point_z.set("0")
        
        # 更新界面
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()

    def add_segment(self):
        """添加线段操作（修正后）"""
        start = self.start_point.get()
        end = self.end_point.get()
        color = self.segment_color
        linestyle = self.linestyle_var.get()
        
        if not start or not end:
            messagebox.showerror("错误", "请选择起点和终点")
            return
        
        success, msg = self.analyzer.add_segment(start, end, color, linestyle)
        if not success:
            messagebox.showerror("错误", msg)
        else:
            messagebox.showinfo("成功", msg)
        
        # 更新界面
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()

    def analyze_geometry(self):
        """分析几何关系（修正后）"""
        if not self.analyzer.segments:
            messagebox.showinfo("提示", "请先添加至少两条线段进行分析")
            return
            
        relations = self.analyzer.analyze_relations()
        
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        # 设置标题
        self.result_text.tag_configure("title", font=("Helvetica", 12, "bold"))
        self.result_text.tag_configure("normal", font=("Helvetica", 10))
        self.result_text.tag_configure("section", font=("Helvetica", 11, "bold"))
        
        self.result_text.insert(tk.END, "几何关系分析结果\n", "title")
        self.result_text.insert(tk.END, "="*40 + "\n\n", "normal")
        
        # 垂直关系
        if relations['perpendicular']:
            self.result_text.insert(tk.END, "【垂直关系】\n", "section")
            for seg1, seg2 in relations['perpendicular']:
                self.result_text.insert(tk.END, f"  • {seg1} ⊥ {seg2}\n", "normal")
            self.result_text.insert(tk.END, "\n", "normal")
        
        # 平行关系
        if relations['parallel']:
            self.result_text.insert(tk.END, "【平行关系】\n", "section")
            for seg1, seg2 in relations['parallel']:
                self.result_text.insert(tk.END, f"  • {seg1} ∥ {seg2}\n", "normal")
            self.result_text.insert(tk.END, "\n", "normal")
        
        # 长度比关系
        if relations['length_ratio']:
            self.result_text.insert(tk.END, "【长度比例关系】\n", "section")
            for seg1, seg2, ratio in relations['length_ratio']:
                self.result_text.insert(tk.END, f"  • {seg1} : {seg2} = {ratio}:1\n", "normal")
            self.result_text.insert(tk.END, "\n", "normal")
        
        # 长度相等关系
        if relations['length_equal']:
            self.result_text.insert(tk.END, "【相等长度】\n", "section")
            for seg1, seg2 in relations['length_equal']:
                self.result_text.insert(tk.END, f"  • |{seg1}| = |{seg2}|\n", "normal")
            self.result_text.insert(tk.END, "\n", "normal")
        
        # 长度差关系
        if relations['length_diff']:
            self.result_text.insert(tk.END, "【长度差值】\n", "section")
            for seg1, seg2, diff in relations['length_diff']:
                self.result_text.insert(tk.END, f"  • |{seg1}| - |{seg2}| = {diff}\n", "normal")
            self.result_text.insert(tk.END, "\n", "normal")
        
        if not any(relations.values()):
            self.result_text.insert(tk.END, "\n未检测到显著的几何关系", "normal")
        
        self.result_text.config(state=tk.DISABLED)

    def delete_object_action(self):
        """删除点或线段（合并操作）"""
        delete_type = self.delete_type.get()
        obj_name = self.delete_object.get()
        
        if not obj_name:
            messagebox.showerror("错误", "请选择要删除的对象")
            return
        
        if delete_type == "点":
            self.delete_point(obj_name)
        else:
            self.delete_segment(obj_name)

    def delete_point(self, point_name):
        """删除点（修正后）"""
        # 检查是否存在依赖该点的线段
        dependent_segments = [
            seg_name for seg_name, (start, end, color, linestyle) in self.analyzer.segments.items()
            if start == point_name or end == point_name
        ]
        
        if dependent_segments:
            confirm = messagebox.askyesno("确认删除", 
                                       f"点 '{point_name}' 被 {len(dependent_segments)} 条线段引用\n"
                                       f"这些线段是: {', '.join(dependent_segments)}\n"
                                       "删除点将同时删除这些线段，是否继续？")
            if not confirm:
                return
            
            # 删除依赖的线段
            for seg_name in dependent_segments:
                del self.analyzer.segments[seg_name]
        
        # 删除点
        if point_name in self.analyzer.points:
            del self.analyzer.points[point_name]
        else:
            messagebox.showerror("错误", f"点 '{point_name}' 不存在")
            return
        
        # 更新界面
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()
        
        messagebox.showinfo("成功", f"点 '{point_name}' 已删除")

    def delete_segment(self, seg_name):
        """删除线段（修正后）"""
        # 确认删除
        confirm = messagebox.askyesno("确认删除", f"确定删除线段 '{seg_name}' 吗？")
        if not confirm:
            return
        
        # 删除线段
        if seg_name in self.analyzer.segments:
            del self.analyzer.segments[seg_name]
        else:
            messagebox.showerror("错误", "线段不存在")
            return
            
        # 更新界面
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()
        
        messagebox.showinfo("成功", f"线段 '{seg_name}' 已删除")

    def delete_selected_vector(self):
        """删除用户选择的向量"""
        index = self.vector_delete_combo.current()
        if index == -1:
            messagebox.showinfo("提示", "请先选择一个向量")
            return
            
        if self.analyzer.delete_vector(index):
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
            messagebox.showinfo("成功", "向量已删除")
        else:
            messagebox.showerror("错误", "删除向量失败")

    def clear_all_vectors(self):
        """清除所有向量"""
        if not self.analyzer.vectors_to_display:
            messagebox.showinfo("提示", "当前没有向量可清除")
            return
            
        self.analyzer.clear_all_vectors()
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()
        messagebox.showinfo("成功", "所有向量已清除")

    def delete_selected_calculation(self):
        """删除用户选择的计算结果"""
        selection = self.calculation_delete_combo.get()
        if not selection:
            messagebox.showinfo("提示", "请先选择一个计算结果")
            return
            
        # 提取点名称（格式为 "点名称(x, y, z)"）
        point_name = selection.split('(')[0].strip()
        
        # 删除点
        if point_name in self.analyzer.points:
            # 查找并删除与这个点相关的向量
            vectors_to_remove = []
            for i, vec_info in enumerate(self.analyzer.vectors_to_display):
                if vec_info['start'] == point_name or vec_info['end'] == point_name:
                    vectors_to_remove.append(i)
            
            # 从后往前删除向量，避免索引变化
            for i in sorted(vectors_to_remove, reverse=True):
                if i < len(self.analyzer.vectors_to_display):
                    del self.analyzer.vectors_to_display[i]
            
            # 删除点
            del self.analyzer.points[point_name]
            
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
            messagebox.showinfo("成功", f"计算结果点 {point_name} 已删除")
        else:
            messagebox.showerror("错误", f"未找到计算结果点 {point_name}")

    def clear_all_calculations(self):
        """清除所有计算结果"""
        if len([name for name in self.analyzer.points if name.startswith('result_')]) == 0:
            messagebox.showinfo("提示", "当前没有计算结果可清除")
            return
            
        self.analyzer.clear_all_calculations()
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()
        messagebox.showinfo("成功", "所有计算结果已清除")
    
    def delete_function(self):
        name = self.delete_function_name.get()
        if not name:
            messagebox.showerror("错误", "请选择要删除的函数")
            return
        if name in self.analyzer.functions:
            del self.analyzer.functions[name]
            messagebox.showinfo("成功", f"函数 '{name}' 已删除")
            self._update_delete_function_combo()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", "函数不存在")

    def toggle_interaction_mode(self):
        """切换交互模式"""
        self.interaction_mode = not self.interaction_mode
        
        if self.interaction_mode:
            self.btn_interaction.config(text="交互模式: 开启")
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, "交互模式已开启: 点击图形上的点来创建线段\n")
            self.status_text.config(state=tk.DISABLED)
        else:
            self.btn_interaction.config(text="交互模式: 关闭")
            self.selected_points = []  # 清空已选择的点
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, "交互模式已关闭\n")
            self.status_text.config(state=tk.DISABLED)
        
        # 重绘图形以更新点选择状态
        self._redraw_plot()

    def on_plot_click(self, event):
        """处理图形点击事件"""
        if not self.interaction_mode:
            return
            
        if event.inaxes != self.ax:
            return  # 点击在图形外部
            
        # 获取点击坐标
        x, y = event.xdata, event.ydata
        if self.current_view == '3d':
            # 在3D视图中，我们需要找到距离点击位置最近的点
            closest_point = None
            min_dist = float('inf')
            
            # 获取当前的投影对象
            proj = self.ax.get_proj()
            
            # 将3D点转换为屏幕坐标，并找到距离点击位置最近的点
            for name, (px, py, pz) in self.analyzer.points.items():
                # 将3D点转换为屏幕坐标
                x_proj, y_proj, _ = proj3d.proj_transform(px, py, pz, proj)
                
                # 计算屏幕距离
                dist = np.sqrt((x_proj - x)**2 + (y_proj - y)**2)
                
                # 使用非常大的阈值以适应3D视图中的距离值
                if dist < 1000:  # 使用非常大的阈值
                    if dist < min_dist:
                        min_dist = dist
                        closest_point = name
                        
            # 使用非常大的阈值
            if closest_point:
                self.handle_point_selection(closest_point)
        else:
            # 在2D视图中，直接计算距离
            closest_point = None
            min_dist = float('inf')
            
            for name, (px, py, pz) in self.analyzer.points.items():
                dist = np.sqrt((px - x)**2 + (py - y)**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_point = name
            
            if closest_point and min_dist < 0.5:  # 2D视图使用较小的阈值
                self.handle_point_selection(closest_point)

    def handle_point_selection(self, point_name):
        """处理点选择逻辑"""
        # 如果点已被选择，则取消选择
        if point_name in self.selected_points:
            self.selected_points.remove(point_name)
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, f"已取消选择点 '{point_name}'\n")
            self.status_text.config(state=tk.DISABLED)
        else:
            # 添加到选择列表
            self.selected_points.append(point_name)
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, f"已选择点 '{point_name}'\n")
            self.status_text.config(state=tk.DISABLED)
            
            # 如果已选择两个点，则创建线段
            if len(self.selected_points) == 2:
                self.create_segment_from_selection()
        
        # 重绘图形以更新点选择状态
        self._redraw_plot()

    def create_segment_from_selection(self):
        """根据选择的点创建线段"""
        if len(self.selected_points) != 2:
            return
            
        start, end = self.selected_points
        
        # 使用当前选择的颜色和线型
        color = self.segment_color
        linestyle = self.linestyle_var.get()
        
        # 添加线段
        success, msg = self.analyzer.add_segment(start, end, color, linestyle)
        if success:
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, f"成功创建线段: {msg}\n")
            self.status_text.config(state=tk.DISABLED)
        else:
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, f"创建线段失败: {msg}\n")
            self.status_text.config(state=tk.DISABLED)
        
        # 清空选择并更新界面
        self.selected_points = []
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()
    
    # 添加几何作图方法
    def draw_perpendicular(self):
        """绘制垂直线"""
        point = self.perp_point.get()
        segment = self.perp_segment.get()
        
        if not point or not segment:
            messagebox.showerror("错误", "请选择点和线段")
            return
        
        success, msg = self.analyzer.create_perpendicular(point, segment)
        if success:
            messagebox.showinfo("成功", msg)
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", msg)

    def draw_parallel(self):
        """绘制平行线"""
        point = self.parallel_point.get()
        segment = self.parallel_segment.get()
        
        if not point or not segment:
            messagebox.showerror("错误", "请选择点和线段")
            return
        
        success, msg = self.analyzer.create_parallel(point, segment)
        if success:
            messagebox.showinfo("成功", msg)
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", msg)

    def draw_midpoint(self):
        """绘制中点"""
        segment = self.mid_segment.get()
        
        if not segment:
            messagebox.showerror("错误", "请选择线段")
            return
        
        success, msg = self.analyzer.create_midpoint(segment)
        if success:
            messagebox.showinfo("成功", msg)
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", msg)

    def draw_circle(self):
        """绘制圆或球"""
        center = self.circle_center.get()
        radius_seg = self.circle_radius.get()
        
        if not center or not radius_seg:
            messagebox.showerror("错误", "请选择圆心和半径线段")
            return
        
        success, msg = self.analyzer.create_circle_or_sphere(center, radius_seg)
        if success:
            messagebox.showinfo("成功", msg)
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", msg)

    def delete_function(self):
        name = self.delete_function_name.get()
        if not name:
            messagebox.showerror("错误", "请选择要删除的函数")
            return
        if name in self.analyzer.functions:
            del self.analyzer.functions[name]
            messagebox.showinfo("成功", f"函数 '{name}' 已删除")
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", "函数不存在")
    
    def add_function(self):
        name = self.function_name.get().strip()
        expr = self.function_expr.get().strip()
        var = self.function_var.get().strip()
        range_min = self.function_range_min.get().strip()
        range_max = self.function_range_max.get().strip()
        is_3d = self.function_type.get() == '3d'

        if not name or not expr or not var or not range_min or not range_max:
            messagebox.showerror("错误", "请填写所有字段")
            return

        try:
            range_min = float(range_min)
            range_max = float(range_max)
        except ValueError:
            messagebox.showerror("错误", "范围必须为数字")
            return

        if is_3d:
            var_list = var.split(',')
            if len(var_list) != 2:
                messagebox.showerror("错误", "3D函数需要两个变量，如 'x,y'")
                return
            range_vals = (range_min, range_max, range_min, range_max)
        else:
            range_vals = (range_min, range_max)

        success, msg = self.analyzer.add_function(name, expr, var, range_vals, self.segment_color, self.linestyle_var.get(), is_3d)
        if success:
            messagebox.showinfo("成功", msg)
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", msg)
    
    def delete_function(self):
        name = self.delete_function_name.get()
        if not name:
            messagebox.showerror("错误", "请选择要删除的函数")
            return
        if name in self.analyzer.functions:
            del self.analyzer.functions[name]
            messagebox.showinfo("成功", f"函数 '{name}' 已删除")
            self._update_combo_boxes()
            self._update_status()
            self._redraw_plot()
        else:
            messagebox.showerror("错误", "函数不存在")

    def refresh_page(self):
        """刷新页面：仅重绘图形，不清除数据"""
        result = messagebox.askyesno("刷新页面", "确定要刷新当前视图吗？这不会删除任何数据")
        if not result:
            return

        # 仅刷新图形
        self._redraw_plot()

        # 提示用户
        messagebox.showinfo("刷新成功", "图形已刷新，数据未被清除")

    def clear_all_data(self):
        """清除所有几何数据（点、线段、函数、圆/球、向量、计算结果）"""
        result = messagebox.askyesno("清除数据", "确定要清除所有几何数据吗？")
        if not result:
            return

        # 清空分析器中的所有数据
        self.analyzer.points.clear()
        self.analyzer.segments.clear()
        self.analyzer.functions.clear()
        self.analyzer.circles_and_spheres.clear()
        self.analyzer.vectors_to_display.clear()

        # 更新界面
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()

        # 提示用户
        messagebox.showinfo("清除成功", "所有几何数据已清除")

    def reset_page(self):
        """重置页面：清除数据并恢复控件到初始状态"""
        result = messagebox.askyesno("重置页面", "确定要重置整个页面吗？\n这将清除所有数据并恢复默认设置")
        if not result:
            return

        # 1. 清空分析器中的所有数据
        self.analyzer.points.clear()
        self.analyzer.segments.clear()
        self.analyzer.functions.clear()
        self.analyzer.circles_and_spheres.clear()
        self.analyzer.vectors_to_display.clear()

        # 2. 重置左侧控制面板中的输入控件
        # 点添加区域
        self.point_name.set("")
        self.point_x.set("")
        self.point_y.set("")
        self.point_z.set("0")

        # 线段添加区域
        self.start_point.set("")
        self.end_point.set("")
        self.segment_color = '#0000FF'  # 默认蓝色
        self.linestyle_var.set('solid')  # 默认实线

        # 函数输入区域
        self.function_name.set("")
        self.function_expr.set("")
        self.function_var.set("x")
        self.function_range_min.set("0")
        self.function_range_max.set("10")
        self.function_type.set("2d")

        # 删除函数区域
        self.delete_function_name.set("")
        self._update_delete_function_combo()

        # 向量计算区域
        self.vec1_start.set("")
        self.vec1_end.set("")
        self.vec2_start.set("")
        self.vec2_end.set("")
        self.calculation_type.set("加法")

        # 其他删除区域
        self.delete_type.set("点")
        self.delete_object.set("")

        # 3. 更新界面状态
        self._update_combo_boxes()
        self._update_status()
        self._redraw_plot()

        # 4. 提示用户
        messagebox.showinfo("重置成功", "所有数据和控件已恢复到初始状态")

    def toggle_3d_view(self):
        """切换3D/2D视图（修正后）"""
        # 清除当前画布并重新创建子图
        self.fig.clf()
        
        # 创建新视图
        if self.current_view == '2d':
            # 切换到3D
            self.current_view = '3d'
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_facecolor("white")  # 白色背景
            self.ax.set_title("3D几何视图", color="black")
            self.ax.tick_params(colors='black')
            
        else:
            # 切换回2D
            self.current_view = '2d'
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor("white")  # 白色背景
            self.ax.set_title("2D几何视图", color="black")
            self.ax.tick_params(colors='black')
            
        
        # 重绘图形
        self._redraw_plot()

    def _redraw_plot(self):
        """重绘当前视图（2D或3D）"""
        # 清除当前轴内容
        self.ax.clear()

        # 绘制基础几何对象（点、线段）
        if self.current_view == '2d':
            self._draw_2d()
        else:
            self._draw_3d()

        # 绘制所有圆/球
        for name, info in self.analyzer.circles_and_spheres.items():
            center = self.analyzer.points[info['center']]
            radius = info['radius']
            color = info['color']
            linestyle = info['linestyle']

            if self.current_view == '2d':
                # 2D视图绘制圆
                circle = plt.Circle((center[0], center[1]), radius,
                                    fill=False, color=color,
                                    linestyle=linestyle, linewidth=1.5)
                self.ax.add_patch(circle)
                self.ax.text(center[0], center[1] + radius + 0.2, name,
                            fontsize=9, color=color,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            else:
                # 3D视图绘制球
                u = np.linspace(0, 2 * np.pi, 30)
                v = np.linspace(0, np.pi, 30)
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                self.ax.plot_surface(x, y, z, color=color, alpha=0.2, linewidth=0)
                self.ax.text(center[0], center[1], center[2] + radius + 0.3, name,
                            fontsize=9, color=color,
                            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        # 新增：绘制所有函数曲线
        for name, info in self.analyzer.functions.items():
            expr = info['expr']
            var = info['var']
            range_vals = info['range']
            color = info['color']
            linestyle = info['linestyle']
            is_3d = info['is_3d']

            if is_3d:
                if self.current_view == '3d':
                    x, y = sp.symbols(var.split(','))
                    func = sp.lambdify((x, y), sp.sympify(expr), 'numpy')
                    x_vals = np.linspace(range_vals[0], range_vals[1], 50)
                    y_vals = np.linspace(range_vals[2], range_vals[3], 50)
                    X, Y = np.meshgrid(x_vals, y_vals)
                    Z = func(X, Y)
                    self.ax.plot_surface(X, Y, Z, color=color, alpha=0.6, linewidth=0.5, edgecolor='black')
                    self.ax.text(X.mean(), Y.mean(), Z.mean(), name,
                                fontsize=9, color=color,
                                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))
            else:
                if self.current_view == '2d':
                    x = sp.symbols(var)
                    func = sp.lambdify(x, sp.sympify(expr), 'numpy')
                    x_vals = np.linspace(range_vals[0], range_vals[1], 500)
                    y_vals = func(x_vals)
                    self.ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, label=name)
                    self.ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        self.canvas_plot.draw_idle()  # 更新画布

    def _draw_2d(self):
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_title("2D几何视图", color="black")
        self.ax.tick_params(colors='black')
        self.ax.set_aspect('equal')

        # 绘制所有点
        for name, (x, y, z) in self.analyzer.points.items():
            if name in self.selected_points:
                self.ax.plot(x, y, 'ro', markersize=10, fillstyle='none', markeredgewidth=2)
                self.ax.plot(x, y, 'bo', markersize=8)
            else:
                self.ax.plot(x, y, 'bo', markersize=8)
            self.ax.text(x + 0.1, y + 0.1, name, fontsize=10, color='black')

        # 绘制线段
        for seg_name, (start, end, color, linestyle) in self.analyzer.segments.items():
            s_x, s_y, s_z = self.analyzer.points[start]
            e_x, e_y, e_z = self.analyzer.points[end]
            self.ax.plot([s_x, e_x], [s_y, e_y], color=color, linestyle=linestyle, linewidth=1.5)

        # 绘制圆
        for name, info in self.analyzer.circles_and_spheres.items():
            center = self.analyzer.points[info['center']]
            radius = info['radius']
            circle = plt.Circle((center[0], center[1]), radius,
                                fill=False, color=info['color'],
                                linestyle=info['linestyle'], linewidth=1.5)
            self.ax.add_patch(circle)
            self.ax.text(center[0], center[1] + radius + 0.2, name,
                        fontsize=9, color=info['color'],
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2))

        #新增：直接绘制 2D 函数曲线
        for name, info in self.analyzer.functions.items():
            if not info['is_3d']:
                expr = info['expr']
                var = info['var']
                range_vals = info['range']
                color = info['color']
                linestyle = info['linestyle']

                x = sp.symbols(var)
                func = sp.lambdify(x, sp.sympify(expr), 'numpy')
                x_vals = np.linspace(range_vals[0], range_vals[1], 500)
                y_vals = func(x_vals)

                self.ax.plot(x_vals, y_vals, color=color, linestyle=linestyle, label=name)
        self.ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

    def _draw_3d(self):
        self.ax.clear()
        self.ax.set_facecolor("white")
        self.ax.grid(True, linestyle='--', alpha=0.3)
        self.ax.set_title("3D几何视图", color="black")
        self.ax.set_xlabel('X', color='black')
        self.ax.set_ylabel('Y', color='black')
        self.ax.set_zlabel('Z', color='black')
        self.ax.tick_params(colors='black')

        # 绘制所有点
        for name, (x, y, z) in self.analyzer.points.items():
            if name in self.selected_points:
                self.ax.scatter(x, y, z, color='r', s=100, edgecolors='black', linewidths=1.5)
            else:
                self.ax.scatter(x, y, z, color='b', s=50)
            self.ax.text(x + 0.1, y + 0.1, z + 0.1, name, fontsize=10, color='black')

        # 绘制线段
        for seg_name, (start, end, color, linestyle) in self.analyzer.segments.items():
            s_x, s_y, s_z = self.analyzer.points[start]
            e_x, e_y, e_z = self.analyzer.points[end]
            self.ax.plot([s_x, e_x], [s_y, e_y], [s_z, e_z],
                        color=color, linestyle=linestyle, linewidth=1.5)

        # 绘制球
        for name, info in self.analyzer.circles_and_spheres.items():
            center = self.analyzer.points[info['center']]
            radius = info['radius']
            u = np.linspace(0, 2 * np.pi, 30)
            v = np.linspace(0, np.pi, 30)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            self.ax.plot_surface(x, y, z, color=info['color'], alpha=0.2, linewidth=0)

        #  新增：直接绘制 3D 函数曲面
        for name, info in self.analyzer.functions.items():
            if info['is_3d']:
                expr = info['expr']
                var = info['var'].split(',')
                range_vals = info['range']
                color = info['color']
                linestyle = info['linestyle']

                x, y = sp.symbols(var)
                func = sp.lambdify((x, y), sp.sympify(expr), 'numpy')

                x_vals = np.linspace(range_vals[0], range_vals[1], 50)
                y_vals = np.linspace(range_vals[2], range_vals[3], 50)
                X, Y = np.meshgrid(x_vals, y_vals)
                Z = func(X, Y)

                self.ax.plot_surface(X, Y, Z, color=color, alpha=0.6, linewidth=0.5, edgecolor='black')

    def _draw_vectors_2d(self):
        """在2D视图中绘制向量箭头（修复版）"""
        for vec_info in self.analyzer.vectors_to_display:
            start_name = vec_info['start']
            end_name = vec_info['end']
            color = vec_info['color']
            label = vec_info['label']
            
            # 从点字典中获取坐标
            start_coord = self.analyzer.points.get(start_name)
            end_coord = self.analyzer.points.get(end_name)
            
            if start_coord is None or end_coord is None:
                continue  # 如果点不存在，跳过
                
            start_x, start_y = start_coord[0], start_coord[1]
            end_x, end_y = end_coord[0], end_coord[1]
            
            # 计算向量分量
            dx = end_x - start_x
            dy = end_y - start_y
            
            # 绘制箭头
            self.ax.quiver(start_x, start_y, dx, dy, 
                          angles='xy', scale_units='xy', scale=1,
                          color=color, label=label, width=0.005,
                          headwidth=8, headlength=10)

    def _draw_vectors_3d(self):
        """在3D视图中绘制向量箭头（修复版）"""
        for vec_info in self.analyzer.vectors_to_display:
            start_name = vec_info['start']
            end_name = vec_info['end']
            color = vec_info['color']
            label = vec_info['label']
            linewidth = vec_info.get('linewidth', 1.5)
            
            # 从点字典中获取坐标
            start_coord = self.analyzer.points.get(start_name)
            end_coord = self.analyzer.points.get(end_name)
            
            if start_coord is None or end_coord is None:
                continue  # 如果点不存在，跳过
                
            start_x, start_y, start_z = start_coord
            end_x, end_y, end_z = end_coord
            
            dx = end_x - start_x
            dy = end_y - start_y
            dz = end_z - start_z
            
            # 绘制3D箭头
            self.ax.quiver(start_x, start_y, start_z, dx, dy, dz,
                          color=color, label=label, length=0.3,
                          arrow_length_ratio=0.1)

    def calculate_vector(self):
        """执行向量计算（完整修复版）"""
        # 获取向量输入方式（点选择或直接输入）
        use_point_selection = True  # 默认使用点选择
        
        # 尝试获取点选择的向量
        try:
            start1 = self.vec1_start.get()
            end1 = self.vec1_end.get()
            start2 = self.vec2_start.get()
            end2 = self.vec2_end.get()
            
            if not all([start1, end1, start2, end2]):
                raise ValueError("请选择完整的向量起点和终点")
            
            vec1 = self.analyzer.get_vector(start1, end1)
            vec2 = self.analyzer.get_vector(start2, end2)
            
            if vec1 is None or vec2 is None:
                raise ValueError("选择的点不存在")
                
        except Exception as e:
            # 如果点选择失败，尝试使用直接输入的坐标
            try:
                vec1_vals = [float(x.strip()) for x in self.vec1_input.get().split(',')]
                vec2_vals = [float(x.strip()) for x in self.vec2_input.get().split(',')]
                
                if len(vec1_vals) != 3 or len(vec2_vals) != 3:
                    raise ValueError("请输入3维坐标（格式：x,y,z）")
                    
                vec1 = np.array(vec1_vals)
                vec2 = np.array(vec2_vals)
                use_point_selection = False
                
                # 创建一个临时点集来表示向量
                timestamp = int(time.time() * 1000)
                
                # 清除旧的临时点
                self.analyzer.clear_temp_points()
                
                # 创建第一个向量的起点和终点
                start1_name = f"temp_start1_{timestamp}"
                end1_name = f"temp_end1_{timestamp}"
                self.analyzer.add_point(start1_name, 0, 0, 0)
                self.analyzer.add_point(end1_name, vec1[0], vec1[1], vec1[2])
                
                # 创建第二个向量的起点和终点
                start2_name = f"temp_start2_{timestamp}"
                end2_name = f"temp_end2_{timestamp}"
                self.analyzer.add_point(start2_name, 0, 0, 0)
                self.analyzer.add_point(end2_name, vec2[0], vec2[1], vec2[2])
                
                # 更新输入值
                start1 = start1_name
                end1 = end1_name
                start2 = start2_name
                end2 = end2_name
                
            except Exception as e:
                messagebox.showerror("输入错误", f"向量输入无效: {str(e)}")
                return

        # 清除旧的向量显示
        self.analyzer.clear_temp_points()
        self.analyzer.vectors_to_display = []
        
        # 添加原始向量
        self.analyzer.vectors_to_display.append({
            'start': start1,
            'end': end1,
            'color': '#0000FF',
            'label': "向量1"
        })
        
        self.analyzer.vectors_to_display.append({
            'start': start2,
            'end': end2,
            'color': '#00FF00',
            'label': "向量2"
        })
        
        # 确保原点O存在
        if "O" not in self.analyzer.points:
            self.analyzer.add_point("O", 0, 0, 0)
        else:
            # 更新原点坐标确保为(0,0,0)
            self.analyzer.points["O"] = (0.0, 0.0, 0.0)

        # 执行计算
        result = []
        calc_type = self.calc_type.get()
        
        # 创建计算结果点
        timestamp = int(time.time() * 1000)
        
        try:
            if calc_type == "加法":
                result_vec = self.analyzer.vector_add(vec1, vec2)
                result_point = f"result_add_{timestamp}"
                self.analyzer.add_point(result_point, result_vec[0], result_vec[1], result_vec[2])
                
                self.analyzer.vectors_to_display.append({
                    'start': "O",
                    'end': result_point,
                    'color': '#FF0000',
                    'label': "加和结果"
                })
                result.append(f"向量加法结果: ({result_vec[0]:.2f}, {result_vec[1]:.2f}, {result_vec[2]:.2f})")
                
            elif calc_type == "减法":
                result_vec = self.analyzer.vector_subtract(vec1, vec2)
                result_point = f"result_sub_{timestamp}"
                self.analyzer.add_point(result_point, result_vec[0], result_vec[1], result_vec[2])
                
                self.analyzer.vectors_to_display.append({
                    'start': "O",
                    'end': result_point,
                    'color': '#FF5500',
                    'label': "减法结果"
                })
                result.append(f"向量减法结果: ({result_vec[0]:.2f}, {result_vec[1]:.2f}, {result_vec[2]:.2f})")
                
            elif calc_type == "点积":
                dot = self.analyzer.vector_dot(vec1, vec2)
                
                # 在x轴上表示点积结果
                dot_point = f"result_dot_{timestamp}"
                dot_pos = (dot * 0.8, 0, 0)  # 放置在x轴上
                self.analyzer.add_point(dot_point, dot_pos[0], dot_pos[1], dot_pos[2])
                
                self.analyzer.vectors_to_display.append({
                    'start': "O",
                    'end': dot_point,
                    'color': '#FF00FF',
                    'label': f"点积: {dot:.2f}",
                    'marker': 'o',
                    'markersize': 10
                })
                result.append(f"向量点积结果: {dot:.2f}")
                
            elif calc_type == "叉积":
                cross = self.analyzer.vector_cross(vec1, vec2)
                result_point = f"result_cross_{timestamp}"
                self.analyzer.add_point(result_point, cross[0], cross[1], cross[2])
                
                self.analyzer.vectors_to_display.append({
                    'start': "O",
                    'end': result_point,
                    'color': '#9900FF',
                    'label': "叉积结果",
                    'linewidth': 2
                })
                result.append(f"向量叉积结果: ({cross[0]:.2f}, {cross[1]:.2f}, {cross[2]:.2f})")
                
            elif calc_type == "模长(向量1)":
                mag = self.analyzer.vector_magnitude(vec1)
                
                # 在y轴上表示模长
                mag_point = f"result_mag1_{timestamp}"
                mag_pos = (0, mag * 0.8, 0)  # 放置在y轴上
                self.analyzer.add_point(mag_point, mag_pos[0], mag_pos[1], mag_pos[2])
                
                self.analyzer.vectors_to_display.append({
                    'start': "O",
                    'end': mag_point,
                    'color': '#FF7700',
                    'label': f"模长: {mag:.2f}",
                    'marker': 's',
                    'markersize': 8
                })
                result.append(f"向量1模长: {mag:.2f}")
                
            elif calc_type == "模长(向量2)":
                mag = self.analyzer.vector_magnitude(vec2)
                
                # 在z轴上表示模长
                mag_point = f"result_mag2_{timestamp}"
                mag_pos = (0, 0, mag * 0.8)  # 放置在z轴上
                self.analyzer.add_point(mag_point, mag_pos[0], mag_pos[1], mag_pos[2])
                
                self.analyzer.vectors_to_display.append({
                    'start': "O",
                    'end': mag_point,
                    'color': '#0099FF',
                    'label': f"模长: {mag:.2f}",
                    'marker': 'd',
                    'markersize': 8
                })
                result.append(f"向量2模长: {mag:.2f}")
                
            elif calc_type == "夹角":
                angle = self.analyzer.vector_angle(vec1, vec2)
                if angle is None:
                    result.append("无法计算夹角（存在零向量）")
                else:
                    # 在xy平面上画一个扇形表示夹角
                    self._draw_angle_in_plot(vec1, vec2, angle)
                    result.append(f"向量夹角: {angle:.2f}°")
            
            # 显示原始向量信息
            result.insert(0, f"向量1 (起点{start1}→终点{end1}): "
                            f"({vec1[0]:.2f}, {vec1[1]:.2f}, {vec1[2]:.2f})")
            result.insert(1, f"向量2 (起点{start2}→终点{end2}): "
                            f"({vec2[0]:.2f}, {vec2[1]:.2f}, {vec2[2]:.2f})\n")
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")
            return

        # 显示结果
        self.vector_result.config(state=tk.NORMAL)
        self.vector_result.delete(1.0, tk.END)
        self.vector_result.insert(tk.END, "\n".join(result))
        self.vector_result.config(state=tk.DISABLED)

        # 重绘图
        self._redraw_plot()

    def _draw_angle_in_plot(self, vec1, vec2, angle_deg):
        """在图中绘制表示夹角的扇形"""
        # 转换为弧度
        angle_rad = np.radians(angle_deg)
        
        # 创建一个临时点集来表示扇形
        num_points = 20
        theta = np.linspace(0, angle_rad, num_points)
        
        # 规范化向量
        vec1_norm = vec1 / np.linalg.norm(vec1)
        vec2_norm = vec2 / np.linalg.norm(vec2)
        
        # 计算基向量
        if not np.allclose(vec1_norm, vec2_norm):
            # 平面法向量
            normal = np.cross(vec1_norm, vec2_norm)
            
            # 使用Gram-Schmidt方法构造正交基
            basis1 = vec1_norm
            basis2 = vec2_norm - np.dot(vec2_norm, basis1) * basis1
            basis2 /= np.linalg.norm(basis2)
            
            # 生成扇形点
            points = []
            scale = min(np.linalg.norm(vec1), np.linalg.norm(vec2)) * 0.5
            
            for t in theta:
                point = scale * (np.cos(t) * basis1 + np.sin(t) * basis2)
                points.append(point)
            
            # 在3D视图中添加扇形
            if self.current_view == '3d':
                points = np.array(points)
                self.ax.plot(points[:, 0], points[:, 1], points[:, 2], 
                            color='#FF00FF', linewidth=1.5, alpha=0.7)
                
                # 添加角度标签
                mid_idx = num_points // 2
                mid_point = points[mid_idx]
                self.ax.text(mid_point[0], mid_point[1], mid_point[2], 
                           f"{angle_deg:.1f}°", fontsize=9, 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
        else:
            # 向量平行，夹角为0
            if self.current_view == '3d':
                # 在中间位置添加标签
                mid_point = vec1_norm * np.linalg.norm(vec1) * 0.4
                self.ax.text(mid_point[0], mid_point[1], mid_point[2], 
                           f"{angle_deg:.1f}°", fontsize=9, 
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

    def on_close(self):
        """窗口关闭时清理资源"""
        plt.close('all')
        self.root.destroy()

if __name__ == "__main__":
    # 配置matplotlib
    rcParams['font.sans-serif'] = ['Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    rcParams['axes.unicode_minus'] = False
    
    root = tk.Tk()
    app = GeometryGUI(root)
    root.mainloop()
