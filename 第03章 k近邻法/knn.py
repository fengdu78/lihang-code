import numpy as np

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


if __name__=='__main__':
	dataset = np.array([[1,2], [2,3],[4,5]])
	data2 = np.array([[2,3],[5,4],[9,6],[4,7],[8,1],[7,2]])
	kd = kdtree(data2)
	kd.preorder(kd.root)
	print(sorted(data2, key=lambda x:x[0]))
	print(data2.shape[0]//2)
	print(data2[data2.shape[0]//2,:])
	print(data2.sort(axis=1))
