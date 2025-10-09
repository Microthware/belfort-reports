import gspread
from gspread_formatting import *
import schedule
import time
from datetime import datetime, timedelta, date
from oauth2client.service_account import ServiceAccountCredentials

red = CellFormat(textFormat=TextFormat(foregroundColor=Color(1, 0, 0)))  # RGB: Red
green = CellFormat(textFormat=TextFormat(foregroundColor=Color(0, 0.5, 0)))  # RGB: Green


# Authenticate with Google Sheets
def authenticate_google_sheets():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    client = gspread.authorize(creds)
    return client


def save_stock_list(file, stock_list):
    client = authenticate_google_sheets()
    sheet = client.open(file).sheet1  # Open the first sheet
    for stoink in stock_list:
        sheet.append_row(stoink)  # Append row


def update_stocks(file):
    client = authenticate_google_sheets()
    sheet = client.open(file).sheet1  # Open the first sheet
    yesterday = date.today() - timedelta(days=1)
    one_week_ago = date.today() - timedelta(days=7)

    rows = sheet.get_all_values()
    for i, row in enumerate(rows, start=1):  # start=1 to match Google Sheets row index
        if row[0] == "date":
            continue
        row_date = datetime.strptime(row[0], "%m/%d/%Y").date()
        if yesterday == row_date:
            next_day = 8.0
            change = percentage_change(float(row[2]), next_day)
            new_values = [row_date.strftime("%m/%d/%Y"), row[1], row[2], str(next_day), change, row[5], row[6]]  # Modify as needed
            sheet.update(f"A{i}:G{i}", [new_values])  # Adjust range based on your columns
            if change.startswith("-"):
                format_cell_range(sheet, "E" + str(i), red)
            else:
                format_cell_range(sheet,"E" + str(i), green)
            print(f"Updated {row[1]} next day results")
        if one_week_ago == row_date:
            next_week = 8.0
            change = percentage_change(float(row[2]), next_week)
            new_values = [row_date.strftime("%m/%d/%Y"), row[1], row[2], row[3], row[4], str(next_week), change]  # Modify as needed
            sheet.update(f"A{i}:G{i}", [new_values])  # Adjust range based on your columns
            if change.startswith("-"):
                format_cell_range(sheet, "G" + str(i), red)
            else:
                format_cell_range(sheet,"G" + str(i), green)
            print(f"Updated {row[1]} results after a week")


def percentage_change(old, new):
    if old == 0:
        return float('inf')  # Avoid division by zero
    val = ((new - old) / old) * 100
    return f"{round(val, 2)}%"







# scheduling bullsit
# # Schedule the task to run at 7 AM every day
# schedule.every().day.at("07:00").do(update_google_sheet)

# print("Scheduler started. Waiting for 7 AM...")

# while True:
#     schedule.run_pending()
#     time.sleep(60)  # Check every minute