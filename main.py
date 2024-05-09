import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
import datetime as dt
import tkinter as tk
from tkinter import messagebox

# Create the main window
root = tk.Tk()
root.title("Stock Price Predictor")

def format_date(x, pos=None):
    return mdates.num2date(x).strftime('%Y-%m-%d')

def fetch_and_predict():
    stock_symbol = symbol_entry.get().upper()
    try:
        start = dt.datetime.strptime(start_date_entry.get(), '%Y-%m-%d')
        end = dt.datetime.strptime(end_date_entry.get(), '%Y-%m-%d')
    except ValueError:
        messagebox.showerror("Invalid Date", "Please enter dates in YYYY-MM-DD format.")
        return

    data = yf.download(stock_symbol, start=start, end=end)
    if data.empty:
        messagebox.showerror("Data Fetch Error", "No data fetched. Check the stock symbol and date range.")
        return
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    seq_length = 40
    x_train = []
    y_train = []
    for i in range(seq_length, len(scaled_data)):
        x_train.append(scaled_data[i-seq_length:i, 0])
        y_train.append(scaled_data[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=10, batch_size=128)

    prediction = model.predict(x_train)
    prediction = scaler.inverse_transform(prediction).flatten()

    last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
    next_day_prediction = model.predict(last_sequence)
    next_day_prediction = scaler.inverse_transform(next_day_prediction).flatten()

    next_day_label.config(text=f"Predicted price for next day: {next_day_prediction[0]:.2f} GBP")

    plot_frame = tk.Frame(root)
    plot_frame.pack(fill=tk.BOTH, expand=True)

    fig, ax = plt.subplots()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.fmt_xdata = format_date

    dates = mdates.date2num(data.index.to_pydatetime())
    line_real, = ax.plot_date(dates, data['Close'].values, '-', label='Real Price', color='red')
    predicted_dates = dates[seq_length:len(prediction) + seq_length]
    line_predicted, = ax.plot_date(predicted_dates, prediction, '-', label='Predicted Price', color='blue')
    ax.set_xlabel("Year")
    ax.set_ylabel("Amount in GBP")
    ax.legend()

    tooltip = ax.annotate("", xy=(0,0), xytext=(20,20), textcoords="offset points",
                          bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.5),
                          arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    tooltip.set_visible(False)

    def on_hover(event):
        if event.inaxes == ax:
            vis = tooltip.get_visible()
            if line_real.contains(event)[0]:
                ind = line_real.contains(event)[1]["ind"][0]
                x, y = line_real.get_data()
                tooltip.xy = (x[ind], y[ind])
                date = format_date(x[ind])
                closest_index = (np.abs(predicted_dates - x[ind])).argmin()
                pred_price = line_predicted.get_data()[1][closest_index]
                tooltip_text = f"Date: {date}, Real Price: {y[ind]:.2f} GBP, Predicted Price: {pred_price:.2f} GBP"
                tooltip.set_text(tooltip_text)
                tooltip.set_visible(True)

                # Calculate the figure and cursor positions
                x_fig, y_fig = ax.transData.transform((x[ind], y[ind]))
                fig_width, fig_height = fig.canvas.get_width_height()
                
                if x_fig > fig_width / 2:
                    tooltip.xytext = (-180, 0)  # Move to left if past midpoint
                else:
                    tooltip.xytext = (180, 0)  # Keep on right if before midpoint

                fig.canvas.draw_idle()
            else:
                if vis:
                    tooltip.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_hover)

    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

input_frame = tk.Frame(root)
input_frame.pack(pady=20, padx=20)

tk.Label(input_frame, text="Stock Symbol:", font=('Arial', 12)).grid(row=0, column=0, sticky='w')
symbol_entry = tk.Entry(input_frame, font=('Arial', 12), width=10)
symbol_entry.grid(row=0, column=1, padx=5)

tk.Label(input_frame, text="Start Date (YYYY-MM-DD):", font=('Arial', 12)).grid(row=1, column=0, sticky='w')
start_date_entry = tk.Entry(input_frame, font=('Arial', 12), width=10)
start_date_entry.grid(row=1, column=1, padx=5)

tk.Label(input_frame, text="End Date (YYYY-MM-DD):", font=('Arial', 12)).grid(row=2, column=0, sticky='w')
end_date_entry = tk.Entry(input_frame, font=('Arial', 12), width=10)
end_date_entry.grid(row=2, column=1, padx=5)

predict_button = tk.Button(root, text="Predict", command=fetch_and_predict, font=('Arial', 12))
predict_button.pack(pady=20)

next_day_label = tk.Label(root, text="", font=('Arial', 12))
next_day_label.pack(pady=10)

root.mainloop()

