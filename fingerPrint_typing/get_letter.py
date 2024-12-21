def typing(finger1, finger2): # 右手食指到小指1-4 左手食指到小指5-8,去掉无名指则为5-7
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    index = (finger1 - 1) * 7 + finger2 - 1
    if index > 25 :
        print('超出字母范围')
        return None, -1, -1
    else:
        #print(letters[index], end='')
        letter = letters[index]
        return letter, -1, -1