import tkinter as tk
from tkinter import ttk, messagebox
import sympy as sp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

class GeometryGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("智能几何分析器")
        self.root.geometry("1200x800")  # 设置窗口大小
        
        # 数据存储
        self.points = {}  # {点名: (x_val, y_val, z_val)} 存储数值坐标
        self.segments = {}  # {线段名: (起点, 终点)}
        
        # 创建界面布局
        self.create_widgets()
        
        # 初始化matplotlib图形
        self.init_matplotlib()
        
        # 自动添加一个正方体
        self.add_default_cube()

    def create_widgets(self):
        # 创建主框架（左侧控制区+右侧绘图区）
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 左侧控制区（宽度占1/3）
        control_frame = ttk.LabelFrame(main_frame, text="控制面板")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, expand=False, padx=5, pady=5)
        
        # 点管理区域
        ttk.Label(control_frame, text="点管理").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Button(control_frame, text="添加点", command=self.show_add_point_dialog).grid(row=1, column=0, padx=5, pady=2)
        ttk.Button(control_frame, text="查看点", command=self.show_points).grid(row=2, column=0, padx=5, pady=2)
        
        # 线段管理区域
        ttk.Label(control_frame, text="线段管理").grid(row=0, column=1, sticky="w", pady=5)
        ttk.Button(control_frame, text="添加线段", command=self.show_add_segment_dialog).grid(row=1, column=1, padx=5, pady=2)
        ttk.Button(control_frame, text="查看线段", command=self.show_segments).grid(row=2, column=1, padx=5, pady=2)
        
        # 关系分析区域
        ttk.Label(control_frame, text="关系分析").grid(row=0, column=2, sticky="w", pady=5)
        ttk.Button(control_frame, text="分析所有关系", command=self.analyze_relations).grid(row=1, column=2, padx=5, pady=2)
        
        # 右侧绘图区（宽度占2/3）
        plot_frame = ttk.LabelFrame(main_frame, text="几何图形")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 创建matplotlib画布
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def init_matplotlib(self):
        """初始化matplotlib图形设置（隐藏自带坐标轴）"""
        # 隐藏默认坐标轴刻度和标签
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.set_zticklabels([])
        
        # 设置固定视角（确保XYZ轴始终可见）
        self.ax.view_init(elev=20, azim=30)
        
        # 绘制固定长度的XYZ轴（原点在(0,0,0)，长度为1）
        self.draw_fixed_axes()
        
        self.canvas.draw()

    def draw_fixed_axes(self):
        """绘制固定长度的XYZ轴（始终显示为直线）"""
        # X轴（红色，沿x方向，长度1）
        self.ax.plot([0, 1], [0, 0], [0, 0], color='red', linewidth=3, label='X轴')
        self.ax.text(1.1, 0, 0, 'X', color='red', fontsize=14, fontweight='bold')
        
        # Y轴（绿色，沿y方向，长度1）
        self.ax.plot([0, 0], [0, 1], [0, 0], color='green', linewidth=3, label='Y轴')
        self.ax.text(0, 1.1, 0, 'Y', color='green', fontsize=14, fontweight='bold')
        
        # Z轴（蓝色，沿z方向，长度1）
        self.ax.plot([0, 0], [0, 0], [0, 1], color='blue', linewidth=3, label='Z轴')
        self.ax.text(0, 0, 1.1, 'Z', color='blue', fontsize=14, fontweight='bold')

    def update_plot(self):
        """更新matplotlib图形（重新绘制所有元素，保留固定XYZ轴）"""
        # 清除之前的点和线段（保留XYZ轴）
        self.ax.cla()  # 清空当前轴的所有内容
        self.draw_fixed_axes()  # 重新绘制固定XYZ轴
        
        # 绘制所有点
        for name, (x, y, z) in self.points.items():
            # 根据z值判断二维/三维点（z=0为二维）
            if z == 0:
                self.ax.scatter(x, y, 0, c='b', marker='^', s=100, label=f'点{name}（二维）')
                self.ax.text(x, y, 0, name, fontsize=10, color='b')
            else:
                self.ax.scatter(x, y, z, c='r', marker='o', s=100, label=f'点{name}（三维）')
                self.ax.text(x, y, z, name, fontsize=10, color='r')
        
        # 绘制所有线段
        for seg_name, (start_name, end_name) in self.segments.items():
            start = self.points[start_name]
            end = self.points[end_name]
            
            # 根据维度设置颜色（二维蓝色，三维红色）
            if start[2] == 0 and end[2] == 0:
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [0, 0], 'b-', alpha=0.7, label=f'线段{seg_name}（二维）')
            else:
                self.ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 'r-', alpha=0.7, label=f'线段{seg_name}（三维）')
        
        # 调整坐标轴范围（自动适应数据，但保留XYZ轴固定长度）
        all_x = [p[0] for p in self.points.values()]
        all_y = [p[1] for p in self.points.values()]
        all_z = [p[2] for p in self.points.values()]
        
        # 计算数据范围（扩展20%边距）
        x_pad = max(abs(min(all_x)), abs(max(all_x))) * 0.2 if all_x else 0.5
        y_pad = max(abs(min(all_y)), abs(max(all_y))) * 0.2 if all_y else 0.5
        z_pad = max(abs(min(all_z)), abs(max(all_z))) * 0.2 if all_z else 0.5
        
        # 设置坐标轴范围（确保覆盖数据和固定XYZ轴）
        self.ax.set_xlim(min(all_x) - x_pad if all_x else -1, max(all_x) + x_pad if all_x else 1)
        self.ax.set_ylim(min(all_y) - y_pad if all_y else -1, max(all_y) + y_pad if all_y else 1)
        self.ax.set_zlim(min(all_z) - z_pad if all_z else -1, max(all_z) + z_pad if all_z else 1)
        
        # 更新图例（仅显示一次）
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        self.canvas.draw()

    def show_add_point_dialog(self):
        """显示添加点的对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("添加点")
        dialog.geometry("300x200")
        
        ttk.Label(dialog, text="点名称:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        name_entry = ttk.Entry(dialog, width=20)
        name_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="X坐标:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        x_entry = ttk.Entry(dialog, width=20)
        x_entry.grid(row=1, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Y坐标:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        y_entry = ttk.Entry(dialog, width=20)
        y_entry.grid(row=2, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="Z坐标（可选，默认0）:").grid(row=3, column=0, padx=5, pady=5, sticky="w")
        z_entry = ttk.Entry(dialog, width=20)
        z_entry.insert(0, "0")
        z_entry.grid(row=3, column=1, padx=5, pady=5)
        
        def add_point():
            name = name_entry.get().strip()
            x_str = x_entry.get().strip()
            y_str = y_entry.get().strip()
            z_str = z_entry.get().strip()
            
            # 输入验证：名称和X/Y坐标不能为空
            if not name or not x_str or not y_str:
                messagebox.showerror("错误", "名称、X、Y坐标不能为空")
                return
            
            try:
                # 转换坐标为数值（处理符号表达式）
                x = self._convert_to_number(x_str)
                y = self._convert_to_number(y_str)
                z = self._convert_to_number(z_str) if z_str else 0
                
                # 检查是否为无穷大或NaN
                if any(np.isinf([x, y, z])):
                    messagebox.showerror("错误", "坐标不能为无穷大（inf）或非数值（NaN）")
                    return
                
                self.add_point(name, x, y, z)
                dialog.destroy()
            except Exception as e:
                messagebox.showerror("错误", f"添加失败：{str(e)}")
        
        ttk.Button(dialog, text="添加", command=add_point).grid(row=4, column=0, columnspan=2, pady=10)

    def _convert_to_number(self, expr_str: str) -> float:
        """将字符串表达式转换为数值（处理符号表达式和常数）"""
        try:
            # 尝试直接转换为浮点数（处理简单数值）
            return float(expr_str)
        except ValueError:
            # 处理符号表达式（如x+1，但需确保无未定义符号）
            try:
                sym_expr = sp.sympify(expr_str)
                # 检查是否为常数（无自由符号）
                if sym_expr.free_symbols:
                    raise ValueError("表达式包含未定义的符号变量")
                return float(sp.N(sym_expr))
            except sp.SympifyError:
                raise ValueError("无效的符号表达式")
            except Exception as e:
                raise ValueError(f"无法转换为数值：{str(e)}")

    def show_add_segment_dialog(self):
        """显示添加线段的对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("添加线段")
        dialog.geometry("300x150")
        
        ttk.Label(dialog, text="起点名称:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        start_entry = ttk.Entry(dialog, width=20)
        start_entry.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(dialog, text="终点名称:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        end_entry = ttk.Entry(dialog, width=20)
        end_entry.grid(row=1, column=1, padx=5, pady=5)
        
        def add_segment():
            start = start_entry.get().strip()
            end = end_entry.get().strip()
            
            # 输入验证：起点和终点名称不能为空
            if not start or not end:
                messagebox.showerror("错误", "起点、终点名称不能为空")
                return
            
            # 验证起点和终点是否存在
            if start not in self.points or end not in self.points:
                messagebox.showerror("错误", "起点或终点不存在")
                return
            
            seg_name = f"{start}{end}"
            if seg_name in self.segments:
                messagebox.showerror("错误", f"线段 '{seg_name}' 已存在")
                return
            
            self.add_segment(start, end)
            dialog.destroy()
        
        ttk.Button(dialog, text="添加", command=add_segment).grid(row=2, column=0, columnspan=2, pady=10)

    def add_point(self, name: str, x: float, y: float, z: float, auto_update=True, ignore_exists=False):
        """添加点（内部方法）"""
        if name in self.points:
            if ignore_exists:
                return
            messagebox.showerror("错误", f"点 '{name}' 已存在")
            return
            
        self.points[name] = (x, y, z)
        messagebox.showinfo("成功", f"添加点 '{name}' 成功")
        if auto_update:
            self.update_plot()  # 更新绘图

    def add_segment(self, start_name: str, end_name: str, auto_update=True, ignore_exists=False):
        """添加线段（内部方法）"""
        if start_name not in self.points or end_name not in self.points:
            if ignore_exists:
                return
            messagebox.showerror("错误", "起点或终点不存在")
            return
            
        seg_name = f"{start_name}{end_name}"
        if seg_name in self.segments:
            if ignore_exists:
                return
            messagebox.showerror("错误", f"线段 '{seg_name}' 已存在")
            return
            
        self.segments[seg_name] = (start_name, end_name)
        messagebox.showinfo("成功", f"添加线段 '{seg_name}' 成功")
        if auto_update:
            self.update_plot()  # 更新绘图

    def add_default_cube(self):
        """添加默认的正方体"""
        # 定义正方体的8个顶点
        cube_points = {
            'A': (0, 0, 0),
            'B': (1, 0, 0),
            'C': (1, 1, 0),
            'D': (0, 1, 0),
            'E': (0, 0, 1),
            'F': (1, 0, 1),
            'G': (1, 1, 1),
            'H': (0, 1, 1)
        }
        
        # 定义正方体的12条边
        edges = [
            ('AB', 'A', 'B'),
            ('BC', 'B', 'C'),
            ('CD', 'C', 'D'),
            ('DA', 'D', 'A'),
            ('AE', 'A', 'E'),
            ('BF', 'B', 'F'),
            ('CG', 'C', 'G'),
            ('DH', 'D', 'H'),
            ('EF', 'E', 'F'),
            ('FG', 'F', 'G'),
            ('GH', 'G', 'H'),
            ('HE', 'H', 'E')
        ]
        
        # 添加顶点
        for name, (x, y, z) in cube_points.items():
            self.add_point(name, x, y, z, auto_update=False, ignore_exists=True)
        
        # 添加边
        for seg_name, start, end in edges:
            self.add_segment(start, end, auto_update=False, ignore_exists=True)
        
        # 更新图形
        self.update_plot()

    def show_points(self):
        """显示所有点信息"""
        if not self.points:
            messagebox.showinfo("点列表", "当前没有点")
            return
            
        info = "当前点列表：\n"
        for name, (x, y, z) in self.points.items():
            info += f"  {name}: ({x:.2f}, {y:.2f}, {z:.2f}) → {'三维' if z != 0 else '二维'}\n"
        
        messagebox.showinfo("点列表", info)

    def show_segments(self):
        """显示所有线段信息"""
        if not self.segments:
            messagebox.showinfo("线段列表", "当前没有线段")
            return
            
        info = "当前线段列表：\n"
        for name, (start, end) in self.segments.items():
            info += f"  {name}: {start}→{end}\n"
        
        messagebox.showinfo("线段列表", info)

    def analyze_relations(self):
        """分析所有线段关系并显示"""
        if len(self.segments) < 2:
            messagebox.showinfo("关系分析", "至少需要2条线段才能分析")
            return
            
        # 计算所有线段的向量
        vectors = {}
        for seg_name, (start_name, end_name) in self.segments.items():
            start = self.points[start_name]
            end = self.points[end_name]
            vectors[seg_name] = (
                end[0] - start[0],  # dx
                end[1] - start[1],  # dy
                end[2] - start[2]   # dz
            )
        
        # 分析关系
        relations = {
            'perpendicular': [],  # 垂直关系
            'parallel': [],       # 平行关系
            'length_ratio': [],   # 长度比
            'length_equal': [],   # 长度相等
            'length_diff': []     # 长度差
        }
        
        seg_names = list(vectors.keys())
        for i in range(len(seg_names)):
            seg1 = seg_names[i]
            v1 = vectors[seg1]
            
            for j in range(i+1, len(seg_names)):
                seg2 = seg_names[j]
                v2 = vectors[seg2]
                
                # 垂直判断（点积为0）
                dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                if abs(dot) < 1e-6:  # 浮点误差容忍
                    relations['perpendicular'].append((seg1, seg2))
                
                # 平行判断（叉积为0）
                cross_x = v1[1]*v2[2] - v1[2]*v2[1]
                cross_y = v1[2]*v2[0] - v1[0]*v2[2]
                cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                cross_mag = (cross_x**2 + cross_y**2 + cross_z**2)**0.5
                if cross_mag < 1e-6:  # 浮点误差容忍
                    relations['parallel'].append((seg1, seg2))
                
                # 长度计算
                len1 = (v1[0]**2 + v1[1]**2 + v1[2]**2)**0.5
                len2 = (v2[0]**2 + v2[1]**2 + v2[2]**2)**0.5
                
                # 长度比
                if len2 != 0:
                    ratio = len1 / len2
                    relations['length_ratio'].append((seg1, seg2, f"{ratio:.2f}"))
                
                # 长度相等
                if abs(len1 - len2) < 1e-6:
                    relations['length_equal'].append((seg1, seg2))
                
                # 长度差
                if len1 != len2:
                    diff = abs(len1 - len2)
                    relations['length_diff'].append((seg1, seg2, f"{diff:.2f}"))
        
        # 显示结果
        result = "线段关系分析结果：\n\n"
        
        if relations['perpendicular']:
            result += "===== 垂直关系 =====\n"
            for seg1, seg2 in relations['perpendicular']:
                result += f"  {seg1} ⊥ {seg2}\n"
        else:
            result += "无垂直关系\n"
        
        if relations['parallel']:
            result += "\n===== 平行关系 =====\n"
            for seg1, seg2 in relations['parallel']:
                result += f"  {seg1} ∥ {seg2}\n"
        else:
            result += "\n无平行关系\n"
        
        if relations['length_ratio']:
            result += "\n===== 长度比关系 =====\n"
            for seg1, seg2, ratio in relations['length_ratio']:
                result += f"  {seg1} 长度 : {seg2} 长度 = {ratio}\n"
        else:
            result += "\n无长度比关系\n"
        
        if relations['length_equal']:
            result += "\n===== 长度相等关系 =====\n"
            for seg1, seg2 in relations['length_equal']:
                result += f"  {seg1} 长度 = {seg2} 长度\n"
        else:
            result += "\n无长度相等关系\n"
        
        if relations['length_diff']:
            result += "\n===== 长度差关系 =====\n"
            for seg1, seg2, diff in relations['length_diff']:
                result += f"  |{seg1} 长度 - {seg2} 长度| = {diff}\n"
        else:
            result += "\n无长度差关系\n"
        
        messagebox.showinfo("关系分析结果", result)

if __name__ == "__main__":
    root = tk.Tk()
    app = GeometryGUI(root)
    root.mainloop()
