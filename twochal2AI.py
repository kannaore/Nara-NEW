import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
from scipy.stats import norm
import tkinter as tk
from tkinter import simpledialog, scrolledtext, messagebox
import sys
import time
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

matplotlib.use('TkAgg')

class Config:
    """설정 정보를 관리하는 클래스"""
    FONT_NAME_NANUM = 'NanumGothic'
    FONT_NAME_DEFAULT = 'sans-serif'
    XMIN = 98
    XMAX = 102
    LINE_COLORS = ['darkblue', 'forestgreen', 'purple', 'red', 'orange', 'teal', 'maroon']
    CLICK_PRECISION_THRESHOLD = 0.1
    
    # 조달청 기준 구간 설정
    PRICE_INTERVALS = [
        (102.000, 101.735), (101.734, 101.469), (101.468, 101.202),
        (101.201, 100.935), (100.934, 100.668), (100.667, 100.401),
        (100.400, 100.134), (100.133, 99.867), (99.866, 99.600),
        (99.599, 99.333), (99.332, 99.066), (99.065, 98.799),
        (98.798, 98.532), (98.531, 98.266), (98.265, 98.000)
    ]

class PriceSimulator:
    """가격 시뮬레이션 관련 기능을 제공하는 클래스"""
    
    @staticmethod
    def generate_official_prices(intervals: List[Tuple[float, float]]) -> List[float]:
        """기초 예비가격 생성"""
        prices = []
        for a, b in intervals:
            low, high = min(a, b), max(a, b)
            price = np.random.uniform(low, high)
            prices.append(price)
        return prices
    
    @staticmethod
    def calculate_avg_prices(pre_prices: List[float], num_combinations: int) -> Tuple[np.ndarray, float, float, float]:
        """조합 평균 계산"""
        sample_size = 4
        avg_prices = []
        for _ in range(num_combinations):
            sampled_prices = np.random.choice(pre_prices, size=sample_size, replace=False)
            avg_prices.append(np.mean(sampled_prices))
        avg_prices_array = np.array(avg_prices)
        mean_avg_prices = np.mean(avg_prices_array)
        std_avg_prices = np.std(avg_prices_array)
        median_avg_price = np.median(avg_prices_array)
        return avg_prices_array, mean_avg_prices, std_avg_prices, median_avg_price
    
    @staticmethod
    def select_and_average(prices: List[float], sample_size: int = 4) -> Tuple[np.ndarray, float]:
        """랜덤 샘플링 후 평균 계산"""
        selected = np.random.choice(prices, size=sample_size, replace=False)
        avg_price = np.mean(selected)
        return selected, avg_price


class BidPriceApp:
    """투찰율 확률 계산 프로그램 메인 클래스"""
    
    def __init__(self):
        """앱 초기화"""
        self.config = Config()
        self.simulator = PriceSimulator()
        
        # 상태 변수들
        self.current_action = None
        self.line_prices = []
        self.avg_prices = None
        self.price_label_texts = []
        self.prob_label_texts = []
        self.status_text = None
        self.fig = None
        self.ax = None
        self.line_labels = []
        self.output_text = None
        self.results_collapsed = False
        self.root = None
        self.output_frame = None
        self.canvas_widget = None
        self.results_button = None
        self.line_color_index = 0
        
        # 폰트 설정
        self._setup_fonts()
    
    def _setup_fonts(self) -> None:
        """폰트 설정 처리"""
        font_path = fm.findfont(self.config.FONT_NAME_NANUM)
        if font_path:
            self.font_name = self.config.FONT_NAME_NANUM
            print(f"폰트 '{self.config.FONT_NAME_NANUM}'을 사용합니다: {font_path}")
        else:
            font_path = self.config.FONT_NAME_DEFAULT
            self.font_name = self.config.FONT_NAME_DEFAULT
            print(f"'{self.config.FONT_NAME_NANUM}' 폰트를 찾을 수 없어 기본 폰트 '{self.config.FONT_NAME_DEFAULT}'을 사용합니다.")
        
        plt.rc('font', family=self.font_name)
        self.font_path = font_path
    
    def format_price(self, price: float) -> str:
        """가격 형식화 (소수점 처리)"""
        return f"{price:.3f}".rstrip('0').rstrip('.')
    
    def draw_distribution(self, mean_avg_prices: float, std_avg_prices: float) -> None:
        """분포도 그리기"""
        if self.avg_prices is None or not self.avg_prices.size:
            print("오류: 시뮬레이션 결과 데이터가 유효하지 않습니다.")
            return

        self.ax.cla()
        self.ax.hist(self.avg_prices, bins=50, density=True, alpha=0.5, color='skyblue', label='예상 예정가격 분포')
        x = np.linspace(self.config.XMIN, self.config.XMAX, 100)
        p = norm.pdf(x, mean_avg_prices, std_avg_prices)
        self.ax.plot(x, p, linewidth=2, color='navy', label='정규분포 근사')

        self.price_label_texts = []
        self.prob_label_texts = []
        probabilities = []
        prev_price = self.config.XMIN
        prev_price_prob = norm.cdf(self.config.XMIN, mean_avg_prices, std_avg_prices)

        price_info_text_lines = []

        sorted_line_prices = sorted(self.line_prices, key=lambda item: item['price'])

        for i, price_info in enumerate(sorted_line_prices):
            price = price_info['price']
            label_name = price_info['label']
            color = price_info['color']

            self.ax.axvline(x=price, color=color, linestyle='-', linewidth=1.5)
            text_y_position = self.ax.get_ylim()[1] * 1.04
            name_text_obj = self.ax.text(price, text_y_position, label_name, ha='center', va='bottom',
                               color='black', fontsize=10, picker=True)
            price_text_obj = self.ax.text(price, text_y_position - self.ax.get_ylim()[1] * 0.01, self.format_price(price),
                               ha='center', va='top', color='black', fontsize=10, picker=True)
            self.price_label_texts.append(name_text_obj)
            self.price_label_texts.append(price_text_obj)

            x_fill = np.linspace(prev_price, price, 100)
            y_fill = norm.pdf(x_fill, mean_avg_prices, std_avg_prices)
            self.ax.fill_between(x_fill, y_fill, color=color, alpha=0.2, hatch='///', edgecolor=color, linewidth=0.5)

            current_price_prob = norm.cdf(price, mean_avg_prices, std_avg_prices)
            prob_area = current_price_prob - prev_price_prob
            probabilities.append(prob_area)
            prob_text = f'{prob_area * 100:.2f}%'
            region_center_x = (prev_price + price) / 2
            region_center_y = self.ax.get_ylim()[1] * 0.25
            prob_text_obj = self.ax.text(region_center_x, region_center_y, prob_text, ha='center', va='center',
                                    color='black', fontsize=10, fontweight='bold')
            self.prob_label_texts.append(prob_text_obj)

            price_info_text_lines.append((color, f"{price_info['label']}: {self.format_price(price)}  {prob_area * 100:.2f}%"))

            prev_price = price
            prev_price_prob = current_price_prob

        # 마지막 영역 처리
        x_fill_last = np.linspace(prev_price, self.config.XMAX, 100)
        y_fill_last = norm.pdf(x_fill_last, mean_avg_prices, std_avg_prices)
        self.ax.fill_between(x_fill_last, y_fill_last, color='lightgray', alpha=0.2, 
                        hatch='///', edgecolor='gray', linewidth=0.5, label='기타 확률')
        last_area_prob = 1 - prev_price_prob
        probabilities.append(last_area_prob)
        prob_text_last = f'{last_area_prob * 100:.2f}%'
        region_center_x_last = (prev_price + self.config.XMAX) / 2
        region_center_y_last = self.ax.get_ylim()[1] * 0.25
        self.ax.text(region_center_x_last, region_center_y_last, prob_text_last, 
                ha='center', va='center', color='black', fontsize=10, fontweight='bold')

        # 그래프 설정
        self.ax.set_title('예정가격 (조합 평균) 분포', fontsize=16, y=1.08, fontfamily=self.font_name)
        self.ax.set_xlabel('예정가격 (조합 평균)', fontsize=12, fontfamily=self.font_name)
        self.ax.set_ylabel('확률 밀도', fontsize=12, fontfamily=self.font_name)
        self.ax.grid(axis='y', linestyle='--', alpha=0.7)
        self.ax.set_xlim(self.config.XMIN, self.config.XMAX)
        self.ax.legend(loc='upper left', prop=fm.FontProperties(fname=self.font_path))
        
        # 정보 박스 표시
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                         alpha=0.3, edgecolor='darkgray', linewidth=0.8)
        start_y_pos = 0.80
        line_height = 0.05

        for i, (color, text) in enumerate(price_info_text_lines):
            y_pos = start_y_pos - i * line_height
            self.ax.text(0.03, y_pos, text, transform=self.ax.transAxes, fontsize=10, 
                    verticalalignment='top', bbox=bbox_props, fontfamily=self.font_name, color=color)

        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.85, left=0.13, right=0.87, bottom=0.1)
        self.fig.canvas.draw_idle()
    
    def display_simulation_results(self, num_combinations: int, mean_avg_prices: float, 
                                 std_avg_prices: float, median_avg_price: float) -> None:
        """시뮬레이션 결과 표시"""
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "[시뮬레이션 결과 분석]\n")
        self.output_text.insert(tk.END, f"- 시뮬레이션 횟수: {num_combinations:,}\n")
        self.output_text.insert(tk.END, f"- 시뮬레이션 평균: {mean_avg_prices:.3f}\n")
        self.output_text.insert(tk.END, f"- 시뮬레이션 표준편차: {std_avg_prices:.3f}\n")
        self.output_text.insert(tk.END, f"- 시뮬레이션 중간값: {median_avg_price:.3f}\n")
        prob_at_median = norm.cdf(median_avg_price, mean_avg_prices, std_avg_prices)
        self.output_text.insert(tk.END, f"- 중간값에서의 누적 확률: {prob_at_median * 100:.2f}%\n")
        self.output_text.insert(tk.END,
                           f"- 이론적인 50% 확률 가격 (정규분포): {norm.ppf(0.5, mean_avg_prices, std_avg_prices):.3f}\n")
        self.output_text.insert(tk.END, "-" * 30 + "\n")
        self.output_text.config(state=tk.DISABLED)
        self.results_button.config(text="결과 숨기기", state=tk.NORMAL)
    
    def toggle_results_panel(self) -> None:
        """결과 패널 토글"""
        if self.avg_prices is None:
            messagebox.showerror("오류", "시뮬레이션 결과를 먼저 실행해주세요.")
            return

        self.results_collapsed = not self.results_collapsed
        if self.results_collapsed:
            self.output_frame.grid_forget()
            self.results_button.config(text="결과 보기")
            self.canvas_widget.grid(row=0, column=0, columnspan=4, sticky="nsew")
        else:
            self.canvas_widget.grid(row=0, column=0, columnspan=3, sticky="nsew")
            self.output_frame.grid(row=0, column=3, sticky="nsew")
            self.results_button.config(text="결과 숨기기")
        self.root.update_idletasks()
        self.fig.tight_layout()
        self.draw_distribution(np.mean(self.avg_prices), np.std(self.avg_prices))
    
    def on_closing(self) -> None:
        """윈도우 닫기 이벤트 처리"""
        if messagebox.askokcancel("종료 확인", "프로그램을 종료하시겠습니까?"):
            self.root.destroy()
            self.root.quit()
    
    def start_simulation(self, num_combinations: int) -> None:
        """시뮬레이션 시작"""
        pre_prices = self.simulator.generate_official_prices(self.config.PRICE_INTERVALS)
        avg_prices_result, mean_avg_prices, std_avg_prices, median_avg_price = self.simulator.calculate_avg_prices(
            pre_prices, num_combinations
        )
        self.avg_prices = avg_prices_result
        median_avg_price_rounded = round(median_avg_price, 1)
        initial_color = self.config.LINE_COLORS[self.line_color_index % len(self.config.LINE_COLORS)]
        self.line_prices = [{'price': median_avg_price_rounded, 'label': '중간값', 'color': initial_color}]
        self.line_color_index += 1
        self.draw_distribution(mean_avg_prices, std_avg_prices)
        self.display_simulation_results(num_combinations, mean_avg_prices, std_avg_prices, median_avg_price)
        self.status_text.set("시뮬레이션 완료")
    
    def on_button_add_clicked(self) -> None:
        """가격 추가 버튼 클릭 이벤트"""
        self.current_action = 'add'
        input_price = simpledialog.askfloat("가격 추가", "추가할 가격을 입력하세요:", 
                                          parent=self.root, minvalue=self.config.XMIN, maxvalue=self.config.XMAX)

        if input_price is not None:
            new_price = round(input_price, 3)
            if not any(item['price'] == new_price for item in self.line_prices):
                new_color = self.config.LINE_COLORS[self.line_color_index % len(self.config.LINE_COLORS)]
                self.line_prices.append({'price': new_price, 'label': f'{len(self.line_prices) + 1}번', 'color': new_color})
                self.line_color_index += 1
                self.status_text.set(f"{new_price} 가격 추가 완료")
                self.draw_distribution(np.mean(self.avg_prices), np.std(self.avg_prices))
            else:
                self.status_text.set("동일 가격 존재")
        else:
            self.status_text.set("가격 추가 취소")
        self.current_action = None
    
    def on_button_remove_clicked(self) -> None:
        """가격 제거 버튼 클릭 이벤트"""
        self.status_text.set("가격 제거 모드: 제거할 가격 영역 클릭")
        self.current_action = 'remove_area'
    
    def on_button_change_clicked(self) -> None:
        """이름 변경 버튼 클릭 이벤트"""
        self.status_text.set("이름 변경 모드: 변경할 가격 영역 클릭")
        self.current_action = 'change_area'
    
    def on_canvas_click(self, event) -> None:
        """캔버스 클릭 이벤트 핸들러"""
        if event.inaxes == self.ax:
            clicked_x = event.xdata
            if clicked_x is not None:
                sorted_line_prices = sorted(self.line_prices, key=lambda item: item['price'])
                prev_price = self.config.XMIN
                closest_price_index = -1

                for i, price_info in enumerate(sorted_line_prices):
                    price = price_info['price']
                    if clicked_x >= prev_price and clicked_x <= price:
                        closest_price_index = i
                        break
                    prev_price = price

                if closest_price_index != -1:
                    current_price_info = sorted_line_prices[closest_price_index]
                    if self.current_action == 'remove_area':
                        removed_price_info = sorted_line_prices.pop(closest_price_index)
                        self.line_prices.remove(removed_price_info)
                        removed_price = removed_price_info['price']
                        removed_label = removed_price_info['label']
                        self.status_text.set(f"가격 {self.format_price(removed_price)} ({removed_label}) 제거 완료")
                        self.draw_distribution(np.mean(self.avg_prices), np.std(self.avg_prices))
                    elif self.current_action == 'change_area':
                        current_name = current_price_info['label']
                        new_name = simpledialog.askstring("이름 변경", f"새 이름 입력 (현재 '{current_name}'):", parent=self.root)
                        if new_name:
                            for original_price_info in self.line_prices:
                                if original_price_info['price'] == current_price_info['price']:
                                    original_price_info['label'] = new_name
                                    break
                            self.status_text.set(f"'{current_name}' -> '{new_name}' 이름 변경 완료")
                            self.draw_distribution(np.mean(self.avg_prices), np.std(self.avg_prices))
                        else:
                            self.status_text.set("이름 변경 취소")
                    self.current_action = None
                else:
                    self.status_text.set("영역을 다시 클릭해주세요.")
            else:
                self.status_text.set("그래프 영역을 클릭해주세요.")
    
    def on_pick(self, event):
        """Pick 이벤트 핸들러"""
        pass

    def export_graph(self) -> None:
        """그래프 내보내기 기능"""
        from tkinter import filedialog
        
        file_types = [
            ('PNG 이미지', '*.png'),
            ('JPEG 이미지', '*.jpg'),
            ('PDF 문서', '*.pdf'),
            ('SVG 이미지', '*.svg')
        ]
        
        filename = filedialog.asksaveasfilename(
            title="그래프 저장",
            filetypes=file_types,
            defaultextension=".png"
        )
        
        if filename:
            try:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                self.status_text.set(f"그래프가 성공적으로 저장되었습니다: {filename}")
            except Exception as e:
                messagebox.showerror("저장 오류", f"그래프 저장 중 오류가 발생했습니다: {str(e)}")
    
    def initialize_gui(self) -> None:
        """GUI 초기화 및 이벤트 연결"""
        self.root = tk.Tk()
        self.root.title("투찰율 확률 계산 프로그램_V5")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.fig, self.ax = plt.subplots(figsize=(12, 7))
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, columnspan=3, sticky="nsew")
        self.fig.canvas.mpl_connect('button_press_event', self.on_canvas_click)
        self.fig.canvas.mpl_connect('pick_event', self.on_pick)

        # 상태 표시줄
        status_frame = tk.Frame(self.root)
        status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.status_text = tk.StringVar()
        status_label = tk.Label(status_frame, textvariable=self.status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_text.set("프로그램 시작")

        # 버튼 프레임
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        add_button = tk.Button(button_frame, text="가격 추가", command=self.on_button_add_clicked)
        add_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        remove_button = tk.Button(button_frame, text="가격 제거", command=self.on_button_remove_clicked)
        remove_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        change_button = tk.Button(button_frame, text="이름 변경", command=self.on_button_change_clicked)
        change_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        export_button = tk.Button(button_frame, text="그래프 저장", command=self.export_graph)
        export_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # 출력 프레임
        self.output_frame = tk.Frame(self.root)
        self.output_frame.grid(row=0, column=3, rowspan=4, sticky="nsew")
        self.output_text = scrolledtext.ScrolledText(self.output_frame, wrap=tk.WORD, height=10)
        self.output_text.grid(row=0, column=0, sticky="nsew")
        self.output_text.insert(tk.END, "[시뮬레이션 결과]\n")
        self.output_text.config(state=tk.DISABLED)

        # 결과 버튼
        self.results_collapsed = False
        self.results_button = tk.Button(self.root, text="결과 보기 (시뮬레이션 후 활성화)", 
                                      state=tk.DISABLED, command=self.toggle_results_panel)
        self.results_button.grid(row=4, column=3, sticky="ew")

        # 그리드 설정
        self.root.grid_columnconfigure(0, weight=3)
        self.root.grid_columnconfigure(1, weight=3)
        self.root.grid_columnconfigure(2, weight=3)
        self.root.grid_columnconfigure(3, weight=1)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=0)
        self.root.grid_rowconfigure(4, weight=0)

        # 초기 시뮬레이션 실행
        num_combinations = simpledialog.askinteger("시뮬레이션 설정", "시뮬레이션 횟수를 입력하세요:", 
                                               initialvalue=500000, parent=self.root, minvalue=1000)
        if num_combinations:
            self.start_simulation(num_combinations)
        
        tk.mainloop()


# 메인 실행
if __name__ == '__main__':
    app = BidPriceApp()
    app.initialize_gui()