def typing_mode_1(finger1, finger2): # 右手食指到小指1-4 左手食指到小指5-8,去掉无名指则为5-7
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    index = (finger1 - 1) * 7 + finger2 - 1
    if index > 25 :
        #print('超出字母范围')
        return None, -1, -1
    else:
        #print(letters[index], end='')
        letter = letters[index]
        return letter, -1, -1

def typing_mode_2(finger1, finger2): # 仅右手，食指到小指第一关节1-4，食指到无名指第二关节5-7
    letters = [
        ['a', 'b', 'c', 'd'],
        ['e', 'f', 'g', 'h'],
        ['i', 'j', 'k', 'l'],
        ['m', 'n', 'o', 'p'],
        ['q', 'r', 's', 't'],
        ['u', 'v', 'w', 'x'],
        ['y', 'z', ',', ' ']
    ]
    idx1 = finger1
    idx2 = finger2
    if idx1 < 0 or idx1 >= len(letters) or idx2 < 0 or idx2 >= len(letters[0]):
        return None, -1, -1
    else:
        #print(letters[idx1][idx2], end='')
        letter = letters[idx1][idx2]
        return letter, -1, -1
