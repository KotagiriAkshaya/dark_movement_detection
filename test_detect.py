from detect import process_video

r = process_video("sample.mp4", "test_out.mp4", simulate_dark=True, max_seconds=15)
print(r)