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


class Data():
    def __init__(self,user,time,latitude,longitude,location):
        self.user = user
        self.time = time
        self.latitude = latitude
        self.longitude = longitude
        self.location = location


    # todo
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


def data_handle(xlsx_path="~/Downloads/LV.xlsx"):
    np.random.seed(0)
    # Matrix_Phi = gen_random_matrix_Phi()        # 只生成1个随机矩阵

    '''真实数据集'''
    Path = xlsx_path
    column_names = ['user', 'time', 'latitude', 'longitude', 'location']
    data_type = {'user': np.int, 'time': str, 'latitude': np.float32, 'longitude': np.float32, 'location': str}

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


if __name__ == '__main__':
    st = self_tree(min_lng = -115.36, max_lng = -115.00, min_lat = 35.55, max_lat = 36.35, layers=4, n=4)
    st.show()

    # 读取文件的数据，转换成 data_list
    data_list = df_to_data_list(data_handle("~/Downloads/LV.xlsx"))

    # 将自定义的 data 列表中的元素插入到树中的叶子结点上
    for data in data_list:
        data.self_insert_to_node_list(st.get_node(st.root),st)

    print("exit")

''' 
数据范围
[-115.36, -115.00, 35.55, 36.35]
'''

'''
1. 自定义 step_lat 和 step_lng 的话，只需调整 latitude 和 longitude 以及层数 n 和树的叉数 layers 
2. n 的值应该为平方数，才是规则的分布。 4/9/16/25 叉数，这样分布比较规则。 n 的取值范围
'''
