import numpy as np
from collections import namedtuple

class kdnode:
	def __init__(self, data, l, left, right):
		self.median = data # 存储在节点中的数据点
		self.l = l # 划分的轴
		self.left = left
		self.right = right

class kdtree:
	def __init__(self, dataset):
		self.k = dataset.shape[1] # 数据维度
		self.root = self.create(dataset, 0)

	def create(self, data, l):
		if data.shape[0]==0:
			return None
		data = np.array(sorted(data, key=lambda x : x[l])) # 将第l维作为排序的key
		median_idx = data.shape[0] // 2
		median = data[median_idx, :]
		
		return kdnode(median, 
				l,
				self.create(data[:median_idx], (l+1) % self.k),
				self.create(data[median_idx+1:], (l+1) % self.k)
				)
	
	def preorder(self, node):
		print(node.median)
		if node.left:
			self.preorder(node.left)
		if node.right:
			self.preorder(node.right)

	def inorder(self, node):
		if node.left:
			self.inorder(node.left)
		print(node.median)
		if node.right:
			self.inorder(node.right)

	def find_nearest(self, node, x, w=2):
		# 目标是少计算点之间的距离

		result = namedtuple('result', ['nearest_point', 'distance'])

		'''
		利用kd树求点x的最近邻 (node, x, current_dist)
		过程: 对任意一个当前遍历到的顶点, 
		(1) 向下遍历至最小的叶节点, 作为最近邻;
		(2) 计算点x到当前划分超平面的距离, 若大于current_dist则不需判断兄弟顶点, 否则递归至兄弟顶点;
		(3) 计算到当前顶点的距离, 若小于当前最小(current_dist或兄弟的最小dist), 则更新最近邻
		返回值result(最近邻, 最小距离)
		'''
		def travel(node, x, current_minn=float('inf'), current_p=[0]*self.k, w=2):
			if node is None:
				return result([0]*self.k, float('inf'))

			point = node.median
			idx = node.l

			if x[idx] <= point[idx]:
				temp = travel(node.left, x, current_minn)
				sibling = node.right
			else:
				temp = travel(node.right, x, current_minn)
				sibling = node.left

			if temp.distance < current_minn:
				current_minn = temp.distance
				current_p = temp.nearest_point

			x2plane = abs(x[idx] - point[idx])

			if x2plane <= current_minn:
				temp = travel(sibling, x, current_minn, current_p)

			if temp.distance < current_minn:
				current_minn = temp.distance
				current_p = temp.nearest_point

			p2p = np.sqrt(sum([(point[i] - x[i])**2 for i in range(self.k)]))

			if p2p < current_minn:
				current_minn = p2p
				current_p = point

			return result(current_p, current_minn)

		return travel(node, x, float('inf'), w)


if __name__=='__main__':
	dataset = np.array([[1,2], [2,3],[4,5]])
	data2 = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
	kd = kdtree(data2)

	print('/-------preorder--------/')
	kd.preorder(kd.root)

	print('/-------inorder--------/')
	kd.inorder(kd.root)
	
	print(type(kd.root))

	re = kd.find_nearest(kd.root,np.array([2.1,3.1]))

	print('Nearest point: ', re.nearest_point, ", Distance=", re.distance)

	re = kd.find_nearest(kd.root, np.array([2,4.5]))

	print('Nearest point: ', re.nearest_point, ", Distance=", re.distance)