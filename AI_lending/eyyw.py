def remove_space_hyphen(input_str):
    # Loại bỏ dấu cách và dấu trừ từ chuỗi
    result_str = ''.join(char for char in input_str if char not in [' ', '-'])
    return result_str

# Sử dụng hàm xóa dấu cách và dấu trừ
input_string = "Hello - World!"
output_string = remove_space_hyphen(input_string)

# In ra kết quả
print(f"Chuỗi ban đầu: '{input_string}'")
print(f"Chuỗi sau khi xóa dấu cách và dấu trừ: '{output_string}'")