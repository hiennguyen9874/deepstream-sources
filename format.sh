# format toàn bộ file đuôi .cpp, .hpp, .cu, .c, .h với file cấu hình
find . -regex '.*\.\(cpp\|hpp\|cu\|c\|h\)' -exec clang-format -style=file -i {} \;
