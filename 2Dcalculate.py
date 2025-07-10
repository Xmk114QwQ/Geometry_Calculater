import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Dict, List, Tuple

class Point:
    """表示几何中的点（支持2D/3D坐标）"""
    def __init__(self, name: str, x: sp.Expr, y: sp.Expr, z: sp.Expr = 0):
        self.name = name  # 点的唯一名称（如"A", "B"）
        self.x = x        # x坐标（数值或符号表达式）
        self.y = y        # y坐标（数值或符号表达式）
        self.z = z        # z坐标（默认0，2D点）
    
    def coords(self) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """返回点的坐标元组（x, y, z）"""
        return (self.x, self.y, self.z)
    
    def is_3d(self) -> bool:
        """判断是否为3D点（z坐标非0或非默认值）"""
        return not (self.z == 0 and (isinstance(self.z, sp.Number) and self.z == 0))

class Segment:
    """表示几何中的线段（由两个点定义）"""
    def __init__(self, start: Point, end: Point):
        self.start = start  # 起点（Point对象）
        self.end = end      # 终点（Point对象）
    
    def name(self) -> str:
        """生成线段的唯一名称（起点名称+终点名称，如"AB"）"""
        return f"{self.start.name}{self.end.name}"
    
    def vector(self) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
        """返回线段的向量（终点坐标 - 起点坐标）"""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dz = self.end.z - self.start.z
        return (dx, dy, dz)
    
    def length(self) -> sp.Expr:
        """计算线段的长度（符号表达式）"""
        v = self.vector()
        if self.start.is_3d():
            return sp.sqrt(v[0]**2 + v[1]**2 + v[2]**2)  # 3D长度公式
        else:
            return sp.sqrt(v[0]**2 + v[1]**2)            # 2D长度公式

class GeometryAnalyzer:
    """几何关系自动分析器（核心功能：自动计算所有线段间的关系）"""
    def __init__(self):
        self.points: Dict[str, Point] = {}       # 存储所有点（名称→Point对象）
        self.segments: Dict[str, Segment] = {}   # 存储所有线段（名称→Segment对象）
    
    def add_point(self, name: str, x: sp.Expr, y: sp.Expr, z: sp.Expr = 0):
        """添加新点（名称唯一）"""
        if name in self.points:
            return False, f"错误：点 '{name}' 已存在"
        self.points[name] = Point(name, x, y, z)
        return True, f"成功添加点 '{name}'（坐标：({x}, {y}, {z})）"
    
    def add_segment(self, start_name: str, end_name: str):
        """添加新线段（由两个已存在的点定义）"""
        if start_name not in self.points or end_name not in self.points:
            return False, f"错误：起点 '{start_name}' 或终点 '{end_name}' 不存在"
        seg_name = Segment(self.points[start_name], self.points[end_name]).name()
        if seg_name in self.segments:
            return False, f"错误：线段 '{seg_name}' 已存在"
        self.segments[seg_name] = Segment(self.points[start_name], self.points[end_name])
        return True, f"成功添加线段 '{seg_name}'（{start_name}→{end_name}）"
    
    def list_points(self) -> str:
        """列出所有已添加的点"""
        if not self.points:
            return "当前没有点"
        result = "\n当前点列表："
        for name, point in self.points.items():
            coords = point.coords()
            z_str = f", z={coords[2]}" if not (coords[2] == 0 and (isinstance(coords[2], sp.Number) and coords[2] == 0)) else ""
            result += f"\n  {name}: ({coords[0]}, {coords[1]}{z_str})"
        return result
    
    def list_segments(self) -> str:
        """列出所有已添加的线段"""
        if not self.segments:
            return "当前没有线段"
        result = "\n当前线段列表："
        for name, seg in self.segments.items():
            start = seg.start.name
            end = seg.end.name
            # 处理长度计算（避免符号变量未定义错误）
            try:
                length = seg.length().evalf(subs=self._get_default_subs())
                result += f"\n  {name}: {start}→{end}（长度≈{length:.2f}）"
            except:
                result += f"\n  {name}: {start}→{end}（长度=符号表达式）"
        return result
    
    def analyze_all_relations(self) -> Dict:
        """
        自动分析所有线段间的几何关系（垂直、平行、长度比等）
        返回：包含所有关系的字典
        """
        relations = {
            'perpendicular': [],  # 垂直关系列表
            'parallel': [],       # 平行关系列表
            'length_ratio': [],   # 长度比关系列表
            'length_equal': [],   # 长度相等关系列表
            'length_difference': []  # 长度差关系列表
        }
        
        # 遍历所有线段对（避免重复计算）
        seg_names = list(self.segments.keys())
        for i in range(len(seg_names)):
            seg1_name = seg_names[i]
            seg1 = self.segments[seg1_name]
            for j in range(i+1, len(seg_names)):
                seg2_name = seg_names[j]
                seg2 = self.segments[seg2_name]
                
                # 计算基础向量关系
                v1 = seg1.vector()
                v2 = seg2.vector()
                seg1_len = seg1.length()
                seg2_len = seg2.length()
                
                # 1. 检查垂直关系（点积为0）
                if seg1.start.is_3d() or seg2.start.is_3d():
                    dot = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]
                else:
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                if dot == 0:
                    relations['perpendicular'].append((seg1_name, seg2_name))
                
                # 2. 检查平行关系（叉积为0）
                if seg1.start.is_3d() or seg2.start.is_3d():
                    cross_x = v1[1]*v2[2] - v1[2]*v2[1]
                    cross_y = v1[2]*v2[0] - v1[0]*v2[2]
                    cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                    cross_mag_sq = cross_x**2 + cross_y**2 + cross_z**2
                    if cross_mag_sq == 0:
                        relations['parallel'].append((seg1_name, seg2_name))
                else:
                    cross_z = v1[0]*v2[1] - v1[1]*v2[0]
                    if cross_z == 0:
                        relations['parallel'].append((seg1_name, seg2_name))
                
                # 3. 计算长度比（seg1_len / seg2_len）
                try:
                    ratio = sp.simplify(seg1_len / seg2_len)
                    if ratio != 0 and ratio != sp.oo:  # 排除0和无穷大
                        relations['length_ratio'].append((seg1_name, seg2_name, ratio))
                except:
                    pass  # 长度比无法计算时跳过
                
                # 4. 检查长度相等（seg1_len == seg2_len）
                try:
                    if seg1_len == seg2_len:
                        relations['length_equal'].append((seg1_name, seg2_name))
                except:
                    pass  # 长度相等无法判断时跳过
                
                # 5. 计算长度差（|seg1_len - seg2_len|）
                try:
                    diff = sp.simplify(sp.Abs(seg1_len - seg2_len))
                    if diff != 0:
                        relations['length_difference'].append((seg1_name, seg2_name, diff))
                except:
                    pass  # 长度差无法计算时跳过
        
        return relations
    
    def _get_all_symbols(self) -> List[sp.Symbol]:
        """获取所有点坐标中使用的符号变量（包括嵌套在表达式中的符号）"""
        symbols = set()
        for point in self.points.values():
            for coord in point.coords():
                if isinstance(coord, sp.Symbol):
                    symbols.add(coord)
                elif isinstance(coord, sp.Expr):
                    # 提取表达式中的所有符号变量（递归处理）
                    for term in coord.args:
                        if isinstance(term, sp.Symbol):
                            symbols.add(term)
        return sorted(symbols, key=lambda x: str(x))
    
    def _get_default_subs(self) -> Dict[sp.Symbol, float]:
        """获取符号变量的默认替换值（用于数值计算）"""
        return {s: 1.0 for s in self._get_all_symbols()}
    
    def render_2d(self, ax):
        """渲染所有线段的2D图形（自动更新）"""
        try:
            ax.clear()  # 清空当前图形
            
            # 收集所有点的坐标（忽略z轴）
            points = []
            labels = []
            for name, point in self.points.items():
                x, y, _ = point.coords()
                # 转换为数值（符号变量用默认值1，数值直接转换）
                try:
                    x_val = float(x.evalf(subs=self._get_default_subs()))
                except:
                    x_val = float(x)  # 处理数值类型（如int/float）
                try:
                    y_val = float(y.evalf(subs=self._get_default_subs()))
                except:
                    y_val = float(y)  # 处理数值类型（如int/float）
                points.append((x_val, y_val))
                labels.append(name)
            
            # 绘制点
            for (x, y), name in zip(points, labels):
                ax.plot(x, y, 'o', markersize=10, label=name)
                ax.text(x+0.1, y+0.1, name, fontsize=10)
            
            # 绘制线段
            for seg_name, seg in self.segments.items():
                start = seg.start
                end = seg.end
                try:
                    x1 = float(start.x.evalf(subs=self._get_default_subs()))
                    y1 = float(start.y.evalf(subs=self._get_default_subs()))
                except:
                    x1, y1 = float(start.x), float(start.y)  # 处理数值类型
                try:
                    x2 = float(end.x.evalf(subs=self._get_default_subs()))
                    y2 = float(end.y.evalf(subs=self._get_default_subs()))
                except:
                    x2, y2 = float(end.x), float(end.y)  # 处理数值类型
                ax.plot([x1, x2], [y1, y2], 'k--', alpha=0.5, 
                        label=f'{seg_name} (len≈{seg.length().evalf(subs=self._get_default_subs()):.2f})')
            
            # 设置图形属性
            ax.set_aspect('equal')
            ax.grid(True)
            ax.set_title('2D几何图形（符号变量默认值为1）')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 图例放右侧
            
            return True
            
        except Exception as e:
            print(f"渲染图形失败：{e}")
            return False

# 主应用类
class GeometryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("几何分析器")
        self.root.geometry("1000x600")  # 设置初始窗口大小
        
        # 创建几何分析器
        self.analyzer = GeometryAnalyzer()
        
        # 创建左右分割的UI布局
        self.splitter = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        self.splitter.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        self.left_frame = ttk.Frame(self.splitter, padding="10")
        self.splitter.add(self.left_frame, weight=1)
        
        # 右侧图形面板
        self.right_frame = ttk.Frame(self.splitter)
        self.splitter.add(self.right_frame, weight=3)
        
        # 设置右侧Matplotlib图形
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化默认点
        self.initialize_default_points()
        
        # 创建左侧控制UI
        self.create_controls()
        
        # 渲染初始图形
        self.analyzer.render_2d(self.ax)
        self.canvas.draw()
    
    def initialize_default_points(self):
        # 添加默认点A, B, C, D
        self.analyzer.add_point('A', 0, 0, 0)
        self.analyzer.add_point('B', 0, 1, 0)
        self.analyzer.add_point('C', 1, 0, 0)
        self.analyzer.add_point('D', 1, 1, 0)
        # 添加默认线段
        self.analyzer.add_segment('A', 'B')
        self.analyzer.add_segment('C', 'D')
        self.analyzer.add_segment('A', 'D')
    
    def create_controls(self):
        # 创建选项卡
        notebook = ttk.Notebook(self.left_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 几何操作选项卡
        geometry_frame = ttk.Frame(notebook, padding="10")
        notebook.add(geometry_frame, text="几何操作")
        
        # 分析选项卡
        analysis_frame = ttk.Frame(notebook, padding="10")
        notebook.add(analysis_frame, text="分析")
        
        # 状态选项卡
        status_frame = ttk.Frame(notebook, padding="10")
        notebook.add(status_frame, text="状态")
        
        # ===== 几何操作选项卡 =====
        # 添加点控件
        ttk.Label(geometry_frame, text="添加点", font=("", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # 点名称
        ttk.Label(geometry_frame, text="点名称:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.point_name_var = tk.StringVar()
        ttk.Entry(geometry_frame, textvariable=self.point_name_var, width=15).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        # X坐标
        ttk.Label(geometry_frame, text="X坐标:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.x_var = tk.StringVar()
        ttk.Entry(geometry_frame, textvariable=self.x_var, width=15).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        # Y坐标
        ttk.Label(geometry_frame, text="Y坐标:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.y_var = tk.StringVar()
        ttk.Entry(geometry_frame, textvariable=self.y_var, width=15).grid(row=3, column=1, sticky=tk.W, pady=2)
        
        # Z坐标
        ttk.Label(geometry_frame, text="Z坐标:").grid(row=4, column=0, sticky=tk.W, pady=2)
        self.z_var = tk.StringVar(value="0")
        ttk.Entry(geometry_frame, textvariable=self.z_var, width=15).grid(row=4, column=1, sticky=tk.W, pady=2)
        
        # 添加点按钮
        ttk.Button(geometry_frame, text="添加点", command=self.add_point).grid(row=5, column=0, columnspan=2, pady=10)
        
        # 添加线段控件
        ttk.Label(geometry_frame, text="添加线段", font=("", 12, "bold")).grid(row=0, column=2, columnspan=2, sticky=tk.W, pady=(0, 10))
        
        # 起点选择
        ttk.Label(geometry_frame, text="起点:").grid(row=1, column=2, sticky=tk.W, pady=2)
        self.start_var = tk.StringVar()
        self.start_combo = ttk.Combobox(geometry_frame, textvariable=self.start_var, state="readonly", width=12)
        self.start_combo.grid(row=1, column=3, sticky=tk.W, pady=2)
        
        # 终点选择
        ttk.Label(geometry_frame, text="终点:").grid(row=2, column=2, sticky=tk.W, pady=2)
        self.end_var = tk.StringVar()
        self.end_combo = ttk.Combobox(geometry_frame, textvariable=self.end_var, state="readonly", width=12)
        self.end_combo.grid(row=2, column=3, sticky=tk.W, pady=2)
        
        # 添加线段按钮
        ttk.Button(geometry_frame, text="添加线段", command=self.add_segment).grid(row=3, column=2, columnspan=2, pady=10)
        
        # 更新点列表
        self.update_point_list()
        
        # ===== 分析选项卡 =====
        # 分析按钮
        ttk.Button(analysis_frame, text="分析所有几何关系", 
                  command=self.analyze_relations).pack(pady=(0, 10))
        
        # 显示分析结果
        ttk.Label(analysis_frame, text="分析结果:").pack(anchor=tk.W)
        self.result_text = scrolledtext.ScrolledText(analysis_frame, width=40, height=15)
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # ===== 状态选项卡 =====
        # 显示当前状态
        ttk.Label(status_frame, text="状态信息:", font=("", 12, "bold")).pack(anchor=tk.W)
        self.status_text = scrolledtext.ScrolledText(status_frame, width=40, height=10)
        self.status_text.pack(fill=tk.BOTH, expand=True)
        self.update_status()
    
    def update_point_list(self):
        """更新点列表下拉菜单"""
        points = list(self.analyzer.points.keys())
        self.start_combo['values'] = points
        self.end_combo['values'] = points
        
        # 如果有可选项，设置默认选择第一个
        if points:
            self.start_combo.current(0)
            self.end_combo.current(0)
    
    def add_point(self):
        """添加点操作"""
        name = self.point_name_var.get().strip()
        x_expr = self.x_var.get().strip()
        y_expr = self.y_var.get().strip()
        z_expr = self.z_var.get().strip()
        
        # 验证输入
        if not name:
            messagebox.showerror("错误", "点名称不能为空")
            return
        if not x_expr or not y_expr:
            messagebox.showerror("错误", "X和Y坐标不能为空")
            return
        
        try:
            x = sp.sympify(x_expr)
            y = sp.sympify(y_expr)
            z = sp.sympify(z_expr) if z_expr else sp.sympify('0')
        except Exception as e:
            messagebox.showerror("错误", f"坐标解析失败: {str(e)}")
            return
        
        # 添加点
        success, message = self.analyzer.add_point(name, x, y, z)
        messagebox.showinfo("操作结果", message)
        
        # 清空输入
        self.point_name_var.set("")
        self.x_var.set("")
        self.y_var.set("")
        self.z_var.set("0")
        
        # 更新点列表
        self.update_point_list()
        
        # 更新状态
        self.update_status()
        
        # 重新渲染图形
        self.analyzer.render_2d(self.ax)
        self.canvas.draw()
    
    def add_segment(self):
        """添加线段操作"""
        start = self.start_var.get()
        end = self.end_var.get()
        
        # 验证输入
        if not start or not end:
            messagebox.showerror("错误", "请选择起点和终点")
            return
        
        # 添加线段
        success, message = self.analyzer.add_segment(start, end)
        messagebox.showinfo("操作结果", message)
        
        # 更新状态
        self.update_status()
        
        # 重新渲染图形
        self.analyzer.render_2d(self.ax)
        self.canvas.draw()
    
    def analyze_relations(self):
        """分析几何关系并显示结果"""
        # 清空结果
        self.result_text.delete(1.0, tk.END)
        
        # 获取分析结果
        relations = self.analyzer.analyze_all_relations()
        
        # 显示分析结果
        self.result_text.insert(tk.END, "===== 几何关系分析结果 =====\n\n")
        
        # 垂直关系
        if relations['perpendicular']:
            self.result_text.insert(tk.END, "【垂直关系】\n")
            for seg1, seg2 in relations['perpendicular']:
                self.result_text.insert(tk.END, f"  ● {seg1} ⊥ {seg2}\n")
            self.result_text.insert(tk.END, "\n")
        else:
            self.result_text.insert(tk.END, "【垂直关系】无\n\n")
        
        # 平行关系
        if relations['parallel']:
            self.result_text.insert(tk.END, "【平行关系】\n")
            for seg1, seg2 in relations['parallel']:
                self.result_text.insert(tk.END, f"  ● {seg1} ∥ {seg2}\n")
            self.result_text.insert(tk.END, "\n")
        else:
            self.result_text.insert(tk.END, "【平行关系】无\n\n")
        
        # 长度比关系
        if relations['length_ratio']:
            self.result_text.insert(tk.END, "【长度比关系】\n")
            for seg1, seg2, ratio in relations['length_ratio']:
                self.result_text.insert(tk.END, f"  ● {seg1} 长度 : {seg2} 长度 = {ratio}\n")
            self.result_text.insert(tk.END, "\n")
        else:
            self.result_text.insert(tk.END, "【长度比关系】无\n\n")
        
        # 长度相等关系
        if relations['length_equal']:
            self.result_text.insert(tk.END, "【长度相等关系】\n")
            for seg1, seg2 in relations['length_equal']:
                self.result_text.insert(tk.END, f"  ● {seg1} 长度 = {seg2} 长度\n")
            self.result_text.insert(tk.END, "\n")
        else:
            self.result_text.insert(tk.END, "【长度相等关系】无\n\n")
        
        # 长度差关系
        if relations['length_difference']:
            self.result_text.insert(tk.END, "【长度差关系】\n")
            for seg1, seg2, diff in relations['length_difference']:
                self.result_text.insert(tk.END, f"  ● |{seg1} 长度 - {seg2} 长度| = {diff}\n")
            self.result_text.insert(tk.END, "\n")
        else:
            self.result_text.insert(tk.END, "【长度差关系】无\n\n")
    
    def update_status(self):
        """更新状态信息"""
        status = f"点数量: {len(self.analyzer.points)}  |  线段数量: {len(self.analyzer.segments)}\n"
        status += f"可用符号变量: {', '.join(self.analyzer._get_all_symbols()) if self.analyzer._get_all_symbols() else '无'}"
        self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, status)

# 主程序
def main():
    root = tk.Tk()
    app = GeometryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
