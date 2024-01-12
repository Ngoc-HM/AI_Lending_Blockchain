def compare_strings_ignore_spaces_and_hyphens(str1, str2):
    # Loại bỏ các ký tự không mong muốn (dấu cách và dấu trừ) từ cả hai chuỗi
    cleaned_str1 =''.join(char for char in str1 if char not in [' ', '-'])
    cleaned_str2 =''.join(char for char in str2 if char not in [' ', '-'])

    # So sánh chuỗi đã làm sạch
    return cleaned_str1.lower() == cleaned_str2.lower()
print(compare_strings_ignore_spaces_and_hyphens("aave-v2","AAVE V2"))