import math

import treelib as tl
import numpy as np
import pandas as pd


# 继承 node
class Node(tl.Node):
    def __init__(self,min_lng, min_lat, max_lng, max_lat,tag=None, identifier=None, data=None):
        tl.Node.__init__(self, tag=tag, identifier=identifier, expanded=True, data=data)
        self.step_lng = max_lng - min_lng
        self.step_lat = max_lat - min_lat
        self.min_lng = min_lng
        self.min_lat = min_lat
        self.max_lng = max_lng
        self.max_lat = max_lat
        self.data = [] # 自定义的数据


# 全局变量，用于初始化树的各个结点
node_id = 0


# 自定义树，继承 treelib.Tree
class SelfDefinedTree(tl.Tree):
    def __init__(self):
        tl.Tree.__init__(self)

    def create_nodes_recursion(self,layers, parent, n=4):
        global node_id
        if layers < 1:
            return

        index = math.log(n,2)
        step_lng = parent.step_lng / index
        step_lat = parent.step_lat / index

        nd_list = []
        k = 0
        while k < n:
            min_lng = parent.min_lng + k % index * step_lng
            min_lat = parent.min_lat + k / index * step_lat
            max_lng = min_lng + step_lng
            max_lat = min_lat + step_lat
            nd_list.append(Node(min_lng, min_lat, max_lng,max_lat, tag="T_" \
                                + str(node_id), identifier="N_" + str(node_id), data=None))
            node_id = node_id + 1
            k = k + 1

        for node in nd_list:
            self.add_node(node, parent)
            # 递归创建该结点下的子结点（区域）
            self.create_nodes_recursion(layers - 1, node, n)


    # 算法：根据点来划分边界
    # 输入：一组点
    # 输出：若点为奇数个，那么返回中间点；若点为偶数个，返回重点两点的中点
    def divide_list(self,ls):
        """return a value to divede this list"""
        # print(ls)
        ls.sort()
        # print(ls)
        n = len(ls)
        if n%2==1:
            mid = ls[(n-1)//2]
        else:
            mid =  float(ls[n//2-1]+ls[n//2])/2

        return mid

    def data_list_x_or_y(self,data_list,index):
        dlr = []
        # print("data_list is ", data_list)
        for dl in data_list:
            dlr.append(dl[index])
        return dlr

    def data_list_classifing(self, data_list, tree_node,neighbor_node,divide_type):
        """classify data_list into four groups via tree_node"""
        dlr1 = []
        dlr2 = []
        critical = []
        # print(type(data_list))
        for dl in data_list:
            if dl.longitude < tree_node.max_lng \
               and dl.longitude > tree_node.min_lng \
               and dl.longitude < tree_node.max_lat \
               and dl.longitude > tree_node.min_lat:
                dlr1.append(dl)
            elif dl.longitude < neighbor_node.max_lng \
                and dl.longitude > neighbor_node.min_lng \
                and dl.longitude < neighbor_node.max_lat \
                and dl.longitude > neighbor_node.min_lat:
                dlr2.append(dl)
            # if divide_type == 0 and tree_node.max_lng == neighbor_node.min_lng:
            #     side_dir.append(dl)
            # elif divide_type ==1 and tree_node.max_lat == neighbor_node.min_lng:
            #     side_dir.append(dl)
            else:
                critical.append(dl)
        k = 0
        for c in critical:
            if k == 0:
                dlr1.append(c)
            else:
                dlr2.append(c)
            k = 1 - k

        return dlr1,dlr2


    def node_list_to_tuple_list(self,data_list):
        tl = []
        for nd in data_list:
            tmp = []
            # print(type(nd))
            tmp.append(nd.longitude)
            tmp.append(nd.latitude)
            tl.append(tmp)
        return tl



    def recursion_divide(self,data_list, depth, parent, divide_type=0):
        """handle data_list recursively"""
        global node_id

        if depth == 1:
            # 将自定义的 data 列表中的元素插入到树中的叶子结点上
            for data in data_list:
                parent.data.append(data)
            return

        # data_list = self.node_list_to_tuple_list(data_list_1)
        # get list of data in two dimension
        # x data list

        if divide_type == 0:
            xdl = self.data_list_x_or_y(self.node_list_to_tuple_list(data_list), divide_type)
            middle = self.divide_list(xdl)
        elif divide_type == 1:
            ydl = self.data_list_x_or_y(self.node_list_to_tuple_list(data_list), divide_type)
            middle = self.divide_list(ydl)

        n = 1
        nd_list = []
        while n <= 2:
            if n == 1:
                if divide_type == 0:
                    min_lng = parent.min_lng
                    max_lng = middle
                    min_lat = parent.min_lat
                    max_lat = parent.max_lat
                elif divide_type == 1:
                    min_lng = parent.min_lng
                    max_lng = parent.max_lng
                    min_lat = parent.min_lat
                    max_lat = middle
            elif n == 2:
                if divide_type == 0:
                    min_lng = middle
                    max_lng = parent.max_lng
                    min_lat = parent.min_lat
                    max_lat = parent.max_lat
                elif divide_type == 1:
                    min_lng = parent.min_lng
                    max_lng = parent.max_lng
                    min_lat = middle
                    max_lat = parent.max_lat

            nd_list.append(Node(min_lng, min_lat,max_lng,max_lat, tag="T_" \
                            + str(node_id), identifier = "N_" + str(node_id), data=None))
            node_id = node_id + 1
            n += 1

        # 1,2,3,4
        # 1

        node = nd_list[0]
        neighbor = nd_list[1]
        self.add_node(node,parent)
        self.add_node(neighbor,parent)
        dlr1,dlr2 = self.data_list_classifing(data_list, node, neighbor,divide_type)
        self.recursion_divide(dlr1,depth-1, node,divide_type=1-divide_type)
        self.recursion_divide(dlr2,depth-1, neighbor,divide_type=1-divide_type)


class Data():
    def __init__(self,user,time,latitude,longitude,location):
        self.user = user
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.location = location

    def self_insert_to_node_list(self, insert_node, tree):
        '''自动加入所在的叶子节点的 data_list 中'''
        if(insert_node.is_leaf()):
            insert_node.data.append(self) # 插入具体数据，可以自行如何操作数据
            return

        if self.longitude < insert_node.min_lng or self.longitude > insert_node.max_lng \
                or self.latitude < insert_node.min_lat or self.latitude > insert_node.max_lat:  # 仅测试用，或者过滤数据用
            return

        children = tree.children(insert_node.identifier)
        # 从节点中递归的插入这条数据
        for node in children:
            if self.latitude >= node.min_lat \
                    and self.latitude <= node.max_lat \
                    and self.longitude >= node.min_lng  \
                    and self.longitude <= node.max_lng:
                self.self_insert_to_node_list(node,tree)
                return

    def __hash__(self):
        return hash(self.latitude + self.longitude)

    def __eq__(self,other):
        if self.longitude == other.longitude and self.latitude == other.latitude:
            return True
        else:
            return False


def data_handle(xlsx_path="~/Downloads/LV.xlsx"):
    np.random.seed(0)
    # Matrix_Phi = gen_random_matrix_Phi()        # 只生成1个随机矩阵

    '''真实数据集'''
    Path = xlsx_path
    column_names = ['user', 'time', 'latitude', 'longitude', 'location']
    # data_type = {'user': np.int, 'time': str, 'latitude': np.float32, 'longitude': np.float32, 'location': str}

    xlsx = pd.ExcelFile(Path)
    df = pd.read_excel(xlsx, names=column_names)

    '''剔除计数大于4的位置'''
    location_counts = df.location.value_counts()
    li = list()
    for index, value in location_counts.iteritems():
        if (value > 4):
            li.append(index)
        else:
            break
    df_redu = pd.concat([df.query('location == @i') for i in li])
    df_redu = df_redu.sort_index()
    df_redu = df_redu.reset_index(drop=True)
    df = df_redu
    return df


def df_to_data_list(df):
    '''
    仅针对该数据集
    :param df: 从 xlsx 中获取的 dataframe
    :return: Data List
    '''
    data_list = []
    # 处理数据中的点
    # df = data_handle()
    size = len(df['user'])
    for i in range(0, size):
        data = Data(df['user'][i], df['time'][i], df['latitude'][i],df['longitude'][i], df['location'][i])
        data_list.append(data)

    return data_list


def self_tree(min_lng, min_lat, max_lng,max_lat, layers=2, n =4):
    '''
    构造指定参数的树
    :param min_lng:
    :param min_lat:
    :param max_lng:
    :param max_lat:
    :param layers:
    :param n:
    :return:
    '''
    # 构造边界为 min_lng, min_lat, max_lng, max_lat 的树，step_lng 和 step_lat 根据 layers 自动设定
    global node_id
    root = Node(min_lng, min_lat, max_lng, max_lat, tag="ROOT", identifier="N_"+ str(node_id))
    node_id = node_id + 1
    st = SelfDefinedTree()
    st.add_node(node=root,parent=None)
    st.create_nodes_recursion(layers,root,n)
    return st

def test1():
    st = self_tree(min_lng = -115.36, max_lng = -115.00, min_lat = 35.55, max_lat = 36.35, layers=5, n=4)
    st.show()

    # 读取文件的数据，转换成 data_list
    data_list = df_to_data_list(data_handle("~/Downloads/LV.xlsx"))

    # 将自定义的 data 列表中的元素插入到树中的叶子结点上
    for data in data_list:
        data.self_insert_to_node_list(st.get_node(st.root),st)

    print("exit")


def self_tree1(min_lng, min_lat,max_lng,max_lat, layers=2, data_list=[]):
    global node_id
    node_id = 0
    root = Node(min_lng, min_lat, max_lng, max_lat, tag="ROOT", identifier="N_"+ str(node_id))
    node_id = node_id + 1
    st = SelfDefinedTree()
    st.add_node(node=root, parent=None)
    # data_list_1 = df_to_data_list(data_handle("~/Projects/LV.xls"))
    # print("the data_list_1 is ", data_list_1)

    st.recursion_divide(data_list,layers,root)
    return st

def data_filter(data_list, min_lng, max_lng, min_lat,max_lat):
    new_data_list = []
    for dl in data_list:
        if dl.longitude > max_lng or dl.longitude < min_lng or dl.latitude > max_lat or dl.latitude < min_lat:
            continue
        else:
            new_data_list.append(dl)

    return new_data_list


def test2():

    # 读取文件的数据，转换成 data_list
    data_list = df_to_data_list(data_handle("~/Projects/LV.xlsx"))

    # 数据去重
    data_list = list(set(data_list))

    # 过滤数据
    min_lng = -115.36
    max_lng = -115.00
    min_lat = 35.55
    max_lat = 36.35
    data_list = data_filter(data_list, min_lng,max_lng,min_lat,max_lat)

    st = self_tree1(min_lng = -115.36, max_lng = -115.00, \
                    min_lat = 35.55, max_lat = 36.35, layers=5, data_list=data_list)
    st.show()


    tmp_list = st.nodes["N_7"].data
    print("N_7 结点的数据 list 中第一个结点的 location id： ", tmp_list[0].location)

    print("exit")


if __name__ == '__main__':
    test2()

''' 
数据范围
[-115.36, -115.00, 35.55, 36.35]
'''

'''
1. 自定义 step_lat 和 step_lng 的话，只需调整 latitude 和 longitude 以及层数 n 和树的叉数 layers 
2. n 的值应该为平方数，才是规则的分布。 4/9/16/25 叉数，这样分布比较规则。 n 的取值范围
'''

'''
均分每个区域相通的点数
'''
