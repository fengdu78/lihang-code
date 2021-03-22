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
		data = np.array(sorted(data, key=lambda x : x[l]))
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

	def find_nearest(node, x, w=2):
		
		result = namedtuple('result', ['np', 'nd', 'visited'])

		def travel(node, x, maxd):
			if not node:
				return result([0]*k, float('inf'), 0)

			visited = 1

			cp = node.median
			cl = node.l

			if x[cl] <= cp[cl]:
				first = node.left
				second = node.right
			else:
				first = node.right
				second = node.left

			temp = travel(first, x, maxd)
			
			visited += temp.visited

			nearestp = temp.np
			dist = temp.nd

			if dist < maxd:
				maxd = dist

			if abs(x[cl] - cp[cl]) <= maxd:
				temp = travel(second, x, maxd) 
				visited += temp.visited

			childp = temp.np
			childd = temp.nd

			if childd < maxd:
				maxd = childd
				nearestp = childp

			dist = sqrt(sum((cp[i]-x[i])**w for i in range(k)))

			if dist < maxd:
				maxd = dist
				nearestp = cp

			return result(nearestp, maxd, visited)

		return travel(node, x, float('inf'))

if __name__=='__main__':
	dataset = np.array([[1,2], [2,3],[4,5]])
	data2 = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
	kd = kdtree(data2)
	kd.preorder(kd.root)
	
	print(type(kd.root))

	kd.find_nearest(kd.root,np.array([2,4]))
