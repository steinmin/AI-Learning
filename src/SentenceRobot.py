import random
from datetime import datetime

class SentenceRobot(object):
    def __init__(self):
        self.grammar = dict()
        now = datetime.now()
        random.seed = now.hour * now.minute * 100 + now.second
        return

    # 得到语法字典
    def setGrammarDict(self, gram, linesplit = "\n", gramsplit = "=>"):
        for line in gram.split(linesplit):
            # 去掉首尾空格后，如果为空则退出
            if not line.strip():
                continue
            expr, statement = line.split(gramsplit)
            self.grammar[expr.strip()] = [i.split() for i in statement.split("|")]

    def generate(self, target, isEng = False):
        if target not in self.grammar:
            return target
        find = random.choice(self.grammar[target])
        blank = ' ' if isEng else '' # 如果是英文中间间隔为空格
        return blank.join(self.generate(t, isEng) for t in find)


sr = SentenceRobot()

grammar = '''
战斗 => 施法  ， 结果 。
施法 => 主语 动作 技能
结果 => 主语 获得 效果
主语 => 张飞 | 关羽 | 赵云 | 典韦 | 许褚 | 刘备 | 黄忠 | 曹操 | 鲁班七号 | 貂蝉
动作 => 施放 | 使用 | 召唤
技能 => 一骑当千 | 单刀赴会 | 青龙偃月 | 刀锋铁骑 | 黑暗潜能 | 画地为牢 | 守护机关 | 狂兽血性 | 龙鸣 | 惊雷之龙 | 破云之龙 | 天翔之龙
获得 => 损失 | 获得
效果 => 数值 状态
数值 => 1 | 1000 |5000 | 100
状态 => 法力 | 生命
'''
# sr.setGrammarDict(grammar)
# print(sr.generate("战斗"))
# print(sr.generate("战斗", True))


host = '''
host = 寒暄 ， 报数 询问 具体业务 结尾
报数 = 我是工号 数字 号 ，
数字 = 单个数字 | 数字 单个数字
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ，
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好
询问 = 请问你要 | 您需要
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？'''
# sr.setGrammarDict(host, '\n', '=')
# print(sr.generate('host'))
# print(sr.generate('host'))

# 小明（天天）的爸爸（妈妈），早上好（下午好）！ 我是李（王）老师，您（你）的儿子（女儿）今天（昨天）在学校（操场）闯祸（打人）啦！
teacher = '''
通知 => 称呼 ， 寒暄 ！ 自我介绍 ， 事件描述
称呼 => 学生 的 家长
自我介绍 => 我是 姓 老师
事件描述 => 谁 时间 在学校 地点 行为 啦！
谁 => 称谓 学生性别
时间 => 日期 小时
学生 => 小明 | 天天 | 蛋蛋 | 柱子
家长 => 爸爸 | 妈妈 | 爷爷 | 奶奶 | 外公 | 外婆
寒暄 => 您好 | 早上好 | 下午好 | 晚上好
姓 => 王 | 李 | 赵 | 张 | 欧阳 | 语文 | 数学
称谓 => 你的 | 您的
学生性别 => 儿子 | 女儿
日期 => 今天 | 昨天 | 前天
小时 => 上午 | 下午 | 课间 | 音乐课 | 体育课
地点 => 操场 | 教室 | 厕所 | 食堂
行为 => 骂人 | 打人 | 造谣 | 挨揍 | 罚站
'''
sr.setGrammarDict(teacher)
print(sr.generate('通知'))
print(sr.generate('通知'))