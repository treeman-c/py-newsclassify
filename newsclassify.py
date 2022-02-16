import tkinter as tk
import NewsSpider as ns
import TrainText as tt


#####################################


def urlm1():
    url = t_url.get('0.0', 'end')
    t_text.delete('1.0', 'end')
    try:
        temp = ns.GetText(url)
    except Exception as e:
        t_text.insert('end', '链接错误，请检查')
        return
    t_text.insert('end', temp)
    str_rstx.set(tt.testmodel(temp))


def textm1():
    text = t_text.get('0.0', 'end')
    try:
        classify = tt.testmodel(text)
    except Exception as e:
        str = '分类出现错误' + e.__str__()
        t_text.insert('end', str)
        return
    str_rstx.set(classify)


def norm(event):
    str_rstx.set('未执行')


#####################################


# 将tkinter 对象实例化
win = tk.Tk()
# 设置窗口标题
win.title('文本分类器')
# 设置窗口大小
win.geometry('700x400')
# 进入消息循环（检测到事件，就刷新组件）
# 定义一个label
l_url = tk.Label(win,
                 text='URL',
                 bg='white',
                 font=('Arial', 12),
                 width=5, height=1
                 )
l_url.place(x=10, y=10, anchor='nw')
turl = tk.StringVar()
turl.set('输入链接')
t_url = tk.Text(win, width=30, height=2)
t_url.bind('<Button-1>', norm)
t_url.place(x=10, y=50, anchor='nw')
b_url = tk.Button(win, text='URL内容分类', width=10, height=2, command=urlm1)
b_url.place(x=20, y=150, anchor='nw')
l_text = tk.Label(win,
                  text='文本',
                  bg='white',
                  font=('Arial', 12),
                  width=5, height=1
                  )
l_text.place(x=300, y=10, anchor='nw')
t_text = tk.Text(win, width=50, height=20)
t_text.bind('<B1-Motion>', norm)
t_text.place(x=300, y=50, anchor='nw')
b_text = tk.Button(win, text='文本内容分类', width=10, height=2, command=textm1)
b_text.place(x=150, y=150, anchor='nw')
l_result = tk.Label(win,
                    text='分类结果 ：',
                    bg='white',
                    font=('Arial', 12),
                    width=10, height=1
                    )
l_result.place(x=30, y=300, anchor='nw')
str_rstx = tk.StringVar()
str_rstx.set('未执行')
l_rstx = tk.Label(win,
                  textvariable=str_rstx,
                  bg='white',
                  font=('Arial', 12),
                  width=5, height=1
                  )
l_rstx.place(x=120, y=300, anchor='nw')
win.mainloop()
