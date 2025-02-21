import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm
from scipy.stats import norm
import tkinter as tk
from tkinter import simpledialog, scrolledtext, messagebox
import sys
import time

# --- [폰트 설정] ---
font_name_nanum = 'NanumGothic'
font_name_default = 'sans-serif'

font_path = fm.findfont(font_name_nanum)

if font_path:
    font_name = font_name_nanum
    print(f"폰트 '{font_name_nanum}'을 사용합니다: {font_path}")
else:
    font_path = font_name_default
    font_name = font_name_default
    print(f"'{font_name_nanum}' 폰트를 찾을 수 없어 기본 폰트 '{font_name_default}'을 사용합니다.")


plt.rc('font', family=font_name)

# --- [전역 변수 초기화] ---
current_action = None
line_prices = []
avg_prices = None
price_label_texts = []
prob_label_texts = []
status_text = None
fig = None
ax = None
xmin = 98
xmax = 102
line_labels = []
output_text = None
results_collapsed = False
root = None
output_frame = None
canvas_widget = None
results_button = None
price_info_text_widget = None
line_colors = ['darkblue', 'forestgreen', 'purple', 'red', 'orange', 'teal', 'maroon']
line_color_index = 0
CLICK_PRECISION_THRESHOLD = 0.1

# --- [조달청 기준 구간 설정] ---
intervals = [
    (102.000, 101.735),
    (101.734, 101.469),
    (101.468, 101.202),
    (101.201, 100.935),
    (100.934, 100.668),
    (100.667, 100.401),
    (100.400, 100.134),
    (100.133, 99.867),
    (99.866, 99.600),
    (99.599, 99.333),
    (99.332, 99.066),
    (99.065, 98.799),
    (98.798, 98.532),
    (98.531, 98.266),
    (98.265, 98.000)
]

# --- [1. 기초예비가격 시뮬레이션] ---
def generate_official_prices(intervals):
    prices = []
    for a, b in intervals:
        low, high = min(a, b), max(a, b)
        price = np.random.uniform(low, high)
        prices.append(price)
    return prices

# --- [2. 조합 평균 계산] ---
def calculate_avg_prices(pre_prices, num_combinations):
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

# --- [select_and_average 함수] ---
def select_and_average(prices, sample_size=4):
    selected = np.random.choice(prices, size=sample_size, replace=False)
    avg_price = np.mean(selected)
    return selected, avg_price

# --- [3. 그래프 그리기 함수] ---
def format_price(price):
    return f"{price:.3f}".rstrip('0').rstrip('.')

def draw_distribution(mean_avg_prices, std_avg_prices):
    global ax, avg_prices, line_prices, price_label_texts, prob_label_texts, fig, xmin, xmax

    if avg_prices is None or not avg_prices.size:
        print("오류: 시뮬레이션 결과 데이터가 유효하지 않습니다.")
        return

    ax.cla()
    ax.hist(avg_prices, bins=50, density=True, alpha=0.5, color='skyblue', label='예상 예정가격 분포')
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mean_avg_prices, std_avg_prices)
    ax.plot(x, p, linewidth=2, color='navy', label='정규분포 근사')

    price_label_texts = []
    prob_label_texts = []
    probabilities = []
    prev_price = xmin
    prev_price_prob = norm.cdf(xmin, mean_avg_prices, std_avg_prices)

    price_info_text_lines = []

    sorted_line_prices = sorted(line_prices, key=lambda item: item['price'])

    for i, price_info in enumerate(sorted_line_prices):
        price = price_info['price']
        label_name = price_info['label']
        color = price_info['color']

        ax.axvline(x=price, color=color, linestyle='-', linewidth=1.5)
        text_y_position = ax.get_ylim()[1] * 1.04
        name_text_obj = ax.text(price, text_y_position, label_name, ha='center', va='bottom',
                           color='black', fontsize=10, picker=True)
        # 소수점 3자리까지 표시하되, 불필요한 0 제거
        price_text_obj = ax.text(price, text_y_position - ax.get_ylim()[1] * 0.01, format_price(price),
                           ha='center', va='top', color='black', fontsize=10, picker=True)
        price_label_texts.append(name_text_obj)
        price_label_texts.append(price_text_obj)

        x_fill = np.linspace(prev_price, price, 100)
        y_fill = norm.pdf(x_fill, mean_avg_prices, std_avg_prices)
        ax.fill_between(x_fill, y_fill, color=color, alpha=0.2, hatch='///', edgecolor=color, linewidth=0.5)

        current_price_prob = norm.cdf(price, mean_avg_prices, std_avg_prices)
        prob_area = current_price_prob - prev_price_prob
        probabilities.append(prob_area)
        prob_text = f'{prob_area * 100:.2f}%'
        region_center_x = (prev_price + price) / 2
        region_center_y = ax.get_ylim()[1] * 0.25
        prob_text_obj = ax.text(region_center_x, region_center_y, prob_text, ha='center', va='center',
                                color='black', fontsize=10, fontweight='bold')
        prob_label_texts.append(prob_text_obj)

        price_info_text_lines.append((color, f"{price_info['label']}: {format_price(price)}  {prob_area * 100:.2f}%"))

        prev_price = price
        prev_price_prob = current_price_prob

    # 영역 채우기 및 나머지 코드는 그대로...
    x_fill_last = np.linspace(prev_price, xmax, 100)
    y_fill_last = norm.pdf(x_fill_last, mean_avg_prices, std_avg_prices)
    ax.fill_between(x_fill_last, y_fill_last, color='lightgray', alpha=0.2, hatch='///', edgecolor='gray',
                    linewidth=0.5, label='기타 확률')
    last_area_prob = 1 - prev_price_prob
    probabilities.append(last_area_prob)
    prob_text_last = f'{last_area_prob * 100:.2f}%'
    region_center_x_last = (prev_price + xmax) / 2
    region_center_y_last = ax.get_ylim()[1] * 0.25
    ax.text(region_center_x_last, region_center_y_last, prob_text_last, ha='center', va='center',
            color='black', fontsize=10, fontweight='bold')

    # 타이틀, 축, 범례 등 나머지 설정은 그대로...
    ax.set_title('예정가격 (조합 평균) 분포', fontsize=16, y=1.08, fontfamily=font_name)
    ax.set_xlabel('예정가격 (조합 평균)', fontsize=12, fontfamily=font_name)
    ax.set_ylabel('확률 밀도', fontsize=12, fontfamily=font_name)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_xlim(xmin, xmax)
    ax.legend(loc='upper left', prop=fm.FontProperties(fname=font_path))
    bbox_props = dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3, edgecolor='darkgray', linewidth=0.8)
    start_y_pos = 0.80
    line_height = 0.05
    font_prop = fm.FontProperties(fname=font_path)

    for i, (color, text) in enumerate(price_info_text_lines):
        y_pos = start_y_pos - i * line_height
        ax.text(0.03, y_pos, text, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=bbox_props, fontfamily=font_name, color=color)

    fig.tight_layout()
    fig.subplots_adjust(top=0.85, left=0.13, right=0.87, bottom=0.1)
    fig.canvas.draw_idle()


# --- [4. 시뮬레이션 결과 표시 함수] ---
def display_simulation_results(num_combinations, mean_avg_prices, std_avg_prices, median_avg_price):
    global output_text, results_button

    output_text.config(state=tk.NORMAL)
    output_text.delete(1.0, tk.END)
    output_text.insert(tk.END, "[시뮬레이션 결과 분석]\n")
    output_text.insert(tk.END, f"- 시뮬레이션 횟수: {num_combinations:,}\n")
    output_text.insert(tk.END, f"- 시뮬레이션 평균: {mean_avg_prices:.3f}\n")
    output_text.insert(tk.END, f"- 시뮬레이션 표준편차: {std_avg_prices:.3f}\n")
    output_text.insert(tk.END, f"- 시뮬레이션 중간값: {median_avg_price:.3f}\n")
    prob_at_median = norm.cdf(median_avg_price, mean_avg_prices, std_avg_prices)
    output_text.insert(tk.END, f"- 중간값에서의 누적 확률: {prob_at_median * 100:.2f}%\n")
    output_text.insert(tk.END,
                       f"- 이론적인 50% 확률 가격 (정규분포): {norm.ppf(0.5, mean_avg_prices, std_avg_prices):.3f}\n")
    output_text.insert(tk.END, "-" * 30 + "\n")
    output_text.config(state=tk.DISABLED)
    results_button.config(text="결과 숨기기", state=tk.NORMAL)


# --- [5. 결과 패널 토글 함수] ---
def toggle_results_panel():
    global results_collapsed, output_text, results_button, root, output_frame, canvas_widget, avg_prices

    if avg_prices is None:
        messagebox.showerror("오류", "시뮬레이션 결과를 먼저 실행해주세요.")
        return

    results_collapsed = not results_collapsed
    if results_collapsed:
        output_frame.grid_forget()
        results_button.config(text="결과 보기")
        canvas_widget.grid(row=0, column=0, columnspan=4, sticky="nsew")
    else:
        canvas_widget.grid(row=0, column=0, columnspan=3, sticky="nsew")
        output_frame.grid(row=0, column=3, sticky="nsew")
        results_button.config(text="결과 숨기기")
    root.update_idletasks()
    fig.tight_layout()
    draw_distribution(np.mean(avg_prices), np.std(avg_prices))

# --- [새로운 함수 - 페이드 아웃 메시지 창 표시 함수 (페이드 아웃 제거)] ---
# def show_fading_message(message):
#     global root, canvas_widget
#
#     # --- [메시지 창 (Toplevel) 생성] ---
#     message_box = tk.Toplevel(root)
#     message_box.overrideredirect(True)
#     message_box.attributes('-topmost', True) # 항상 맨 위에 표시
#     message_box.configure(bg=root.cget('bg'))
#
#     # --- [타원형 배경을 위한 Canvas 생성] ---
#     canvas_width = 250
#     canvas_height = 50
#     canvas = tk.Canvas(message_box, width=canvas_width, height=canvas_height, bg=root.cget('bg'), highlightthickness=0)
#     canvas.pack()
#
#     # --- [타원형 배경 그리기] ---
#     oval_bg_color = 'lightyellow'
#     oval_coords = (10, 5, canvas_width - 10, canvas_height - 5)
#     canvas.create_oval(oval_coords, fill=oval_bg_color, outline=oval_bg_color)
#
#     # --- [메시지 레이블 (Canvas 위에 텍스트로 표시)] ---
#     message_label = canvas.create_text(canvas_width / 2, canvas_height / 2, text=message,
#                                        font=('NanumGothic', 12), fill='black')
#
#     # --- [메시지 창 위치 설정 (프로그램 제목 아래)] ---
#     canvas_x = canvas_widget.winfo_x()
#     canvas_y = canvas_widget.winfo_y()
#     canvas_width_widget = canvas_widget.winfo_width()
#     window_width = canvas_width
#     window_height = canvas_height
#
#     x_pos = canvas_x + (canvas_width_widget - window_width) // 2
#     y_pos = canvas_y + 10 # 캔버스 아래 10px 아래에 메시지 창 배치 (타이틀 아래)
#
#     message_box.geometry(f'+{x_pos}+{y_pos}')
#
#     # --- [2초 후에 메시지 창 닫기 (페이드 아웃 효과 제거)] ---
#     root.after(2000, message_box.destroy) # 2000ms = 2초


# --- [윈도우 닫기 버튼 이벤트 핸들러] ---
def on_closing():
    global root
    if messagebox.askokcancel("종료 확인", "프로그램을 종료하시겠습니까?"):
        root.destroy()
        root.quit()


# --- [6. 시뮬레이션 시작 함수] ---
def start_simulation(num_combinations):
    global avg_prices, line_prices, results_button, line_color_index, line_colors, status_text

    pre_prices = generate_official_prices(intervals)
    avg_prices_result, mean_avg_prices, std_avg_prices, median_avg_price = calculate_avg_prices(pre_prices,
                                                                                                num_combinations)
    avg_prices = avg_prices_result
    median_avg_price_rounded = round(median_avg_price, 1)
    initial_color = line_colors[line_color_index % len(line_colors)]
    line_prices = [{'price': median_avg_price_rounded, 'label': '중간값', 'color': initial_color}]
    line_color_index += 1
    draw_distribution(mean_avg_prices, std_avg_prices)
    display_simulation_results(num_combinations, mean_avg_prices, std_avg_prices, median_avg_price)
    status_text.set("시뮬레이션 완료")
    # show_fading_message("시뮬레이션 완료")

# --- [7. 버튼 클릭 이벤트 핸들러 ] ---
def on_button_add_clicked():
    global status_text, current_action, xmin, xmax, line_prices, line_color_index, line_colors
    current_action = 'add'
    input_price = simpledialog.askfloat("가격 추가", "추가할 가격을 입력하세요:", parent=root, minvalue=xmin, maxvalue=xmax)

    if input_price is not None:
        new_price = round(input_price, 3)  # 소수점 3자리로 반올림
        if not any(item['price'] == new_price for item in line_prices):
            new_color = line_colors[line_color_index % len(line_colors)]
            line_prices.append({'price': new_price, 'label': f'{len(line_prices) + 1}번', 'color': new_color})
            line_color_index += 1
            status_text.set(f"{new_price} 가격 추가 완료")
            draw_distribution(np.mean(avg_prices), np.std(avg_prices))
        else:
            status_text.set("동일 가격 존재")
    else:
        status_text.set("가격 추가 취소")
    current_action = None

def on_button_remove_clicked():
    global status_text, current_action
    status_text.set("가격 제거 모드: 제거할 가격 영역 클릭")
    # show_fading_message("가격 제거 모드: 제거할 가격 영역 클릭")
    current_action = 'remove_area'

def on_button_change_clicked():
    global status_text, current_action
    status_text.set("이름 변경 모드: 변경할 가격 영역 클릭")
    # show_fading_message("이름 변경 모드: 변경할 가격 영역 클릭")
    current_action = 'change_area'


# --- [8. 캔버스 클릭 이벤트 핸들러] ---
def on_canvas_click(event):
    global current_action, line_prices, status_text, xmin, xmax, line_labels
    if event.inaxes == ax:
        clicked_x = event.xdata
        if clicked_x is not None:
            closest_price_index = -1
            min_distance = float('inf')

            sorted_line_prices = sorted(line_prices, key=lambda item: item['price'])

            prev_price = xmin

            for i, price_info in enumerate(sorted_line_prices):
                price = price_info['price']
                if clicked_x >= prev_price and clicked_x <= price:
                    closest_price_index = i
                    break
                prev_price = price

            if closest_price_index != -1:
                current_price_info = sorted_line_prices[closest_price_index]
                if current_action == 'remove_area':
                    removed_price_info = sorted_line_prices.pop(closest_price_index)
                    line_prices.remove(removed_price_info)
                    removed_price = removed_price_info['price']
                    removed_label = removed_price_info['label']
                    status_text.set(f"가격 {format_price(removed_price)} ({removed_label}) 제거 완료")
                    draw_distribution(np.mean(avg_prices), np.std(avg_prices))
                elif current_action == 'change_area':
                    current_name = current_price_info['label']
                    new_name = simpledialog.askstring("이름 변경", f"새 이름 입력 (현재 '{current_name}'):",
                                                      parent=root)
                    if new_name:
                        current_price_info['label'] = new_name # sorted_line_prices 는 정렬된 복사본이므로, line_prices 를 직접 수정해야 함
                        for original_price_info in line_prices: # line_prices 에서 해당 가격 정보를 찾아서 이름 변경
                            if original_price_info['price'] == current_price_info['price']:
                                original_price_info['label'] = new_name
                                break
                        status_text.set(f"'{current_name}' -> '{new_name}' 이름 변경 완료")
                        # show_fading_message(f"'{current_name}' -> '{new_name}' 이름 변경 완료")
                        draw_distribution(np.mean(avg_prices), np.std(avg_prices))
                    else:
                        status_text.set("이름 변경 취소")
                        # show_fading_message("이름 변경 취소")
                current_action = None
            else: # '기타 확률' 영역 (마지막 영역) 클릭 시 (현재는 아무 동작 안 함)
                status_text.set("영역을 다시 클릭해주세요.") #  메시지 변경 (더 명확하게)
                # show_fading_message("영역을 다시 클릭해주세요.") # 메시지 변경 (더 명확하게)
        else:
             status_text.set("그래프 영역을 클릭해주세요.")
             # show_fading_message("그래프 영역을 클릭해주세요.")


# --- [9. Pick 이벤트 핸들러] ---
def on_pick(event):
    pass


# --- [10. GUI 초기화 및 이벤트 연결] ---
def initialize_gui():
    global fig, ax, status_text, avg_prices, line_prices, line_labels, output_text, results_button, price_label_texts, price_info_text_widget
    global results_collapsed, root, output_frame, canvas_widget, status_label

    root = tk.Tk()
    root.title("투찰율 확률 계산 프로그램_V5")

    root.protocol("WM_DELETE_WINDOW", on_closing)

    fig, ax = plt.subplots(figsize=(12, 7))
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.grid(row=0, column=0, columnspan=3, sticky="nsew")
    fig.canvas.mpl_connect('button_press_event', on_canvas_click)
    fig.canvas.mpl_connect('pick_event', on_pick)

    status_frame = tk.Frame(root)
    status_frame.grid(row=1, column=0, columnspan=3, sticky="ew")
    status_text = tk.StringVar()
    status_label = tk.Label(status_frame, textvariable=status_text, bd=1, relief=tk.SUNKEN, anchor=tk.W)
    status_label.pack(fill=tk.X, side=tk.BOTTOM)
    status_text.set("프로그램 시작")
    # show_fading_message("프로그램 시작")

    button_frame = tk.Frame(root)
    button_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
    add_button = tk.Button(button_frame, text="가격 추가", command=on_button_add_clicked)
    add_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
    remove_button = tk.Button(button_frame, text="가격 제거", command=on_button_remove_clicked)
    remove_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
    change_button = tk.Button(button_frame, text="이름 변경", command=on_button_change_clicked)
    change_button.pack(side=tk.LEFT, fill=tk.X, expand=True)

    output_frame = tk.Frame(root)
    output_frame.grid(row=0, column=3, rowspan=4, sticky="nsew")
    output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
    output_text.grid(row=0, column=0, sticky="nsew")
    output_text.insert(tk.END, "[시뮬레이션 결과]\n")
    output_text.config(state=tk.DISABLED)

    results_collapsed = False
    results_button = tk.Button(root, text="결과 보기 (시뮬레이션 후 활성화)", state=tk.DISABLED, command=toggle_results_panel)
    results_button.grid(row=4, column=3, sticky="ew")

    root.grid_columnconfigure(0, weight=3)
    root.grid_columnconfigure(1, weight=3)
    root.grid_columnconfigure(2, weight=3)
    root.grid_columnconfigure(3, weight=1)
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=0)
    root.grid_rowconfigure(2, weight=0)
    root.grid_rowconfigure(3, weight=0)
    root.grid_rowconfigure(4, weight=0)


    start_simulation(simpledialog.askinteger("시뮬레이션 설정", "시뮬레이션 횟수를 입력하세요:", initialvalue=500000, parent=root, minvalue=1000))

    tk.mainloop()


# --- [11. 메인 실행] ---
if __name__ == '__main__':
    initialize_gui()