tickets\_gain文件夹反映了8支球队赛年内所有比赛的统计数据以及观赛人数，选取了美国东部和西部各四支球队，各个排名层次均匀抽样，Attend列是比赛参加人数，假设每场比赛票价相同，可以间接反映收入



player\_team\_and\_performance文件夹里的各列含义

Rk : Rank

Player : Player's name

Pos : Position

Age : Player's age

Tm : Team

G : Games played

GS : Games started

MP : Minutes played per game

FG : Field goals per game

FGA : Field goal attempts per game

FG% : Field goal percentage

3P : 3-point field goals per game

3PA : 3-point field goal attempts per game

3P% : 3-point field goal percentage

2P : 2-point field goals per game

2PA : 2-point field goal attempts per game

2P% : 2-point field goal percentage

eFG% : Effective field goal percentage

FT : Free throws per game

FTA : Free throw attempts per game

FT% : Free throw percentage

ORB : Offensive rebounds per game

DRB : Defensive rebounds per game

TRB : Total rebounds per game

AST : Assists per game

STL : Steals per game

BLK : Blocks per game

TOV : Turnovers per game

PF : Personal fouls per game

PTS : Points per game



player\_salary文件夹里的各列含义：

id,Name：球员信息

年份列：该年份的薪资



team\_ranking文件夹包含了两个赛年队伍的总排名和比赛总数据，各列含义如下：

Conference : 联盟归属，分为东部和西部，两个联盟分开计算排名

Regional Rank : 队伍在本联盟的排名

W : wins

L : losses

W/L% : Win rate

GB : Games behind

PS/G : Points per game

PA/G: Opponent points per game

SRG :a team rating that judge the team's ability , zero is the average



player\_social\_influence文件夹记录了球员的影响力数据

All\_star\_player\_voting.csv记录了各年份入选全明星赛的25名球员以及获得的投票数

player\_followers.csv记录了各个球员的Instagram粉丝数



team\_markets文件夹记录了2022年各队的媒体关注者与营收情况

TV Market Size:TV总收视规模，单位为百万人

Metro Population in according city/state：球队所在地区/州的人口总量，单位为百万人

Team Revenue：团队的媒体收入，单位为百万元



team\_salary\_cap文件夹记录了球队每年的薪资上限（salary cap）

payroll : 当年的原始工资帽

inflationAdjPayroll：经过通货膨胀调整后换算到2022-2023赛季的工资帽

