sample_dict = {"content": "abc", "type": "example", "first_historical_data": "2022-01-01"}

# Tạo một danh sách và thêm từ điển vào danh sách
list_of_dicts = [sample_dict]
print("Before append:", list_of_dicts)

# Thay đổi giá trị trong từ điển gốc
sample_dict["content"] = "xyz"

# Thêm lại từ điển vào danh sách
list_of_dicts.append(sample_dict)
print("After append:", list_of_dicts)