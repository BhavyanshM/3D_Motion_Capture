import numpy as np

a = np.array([	[0,-3],
				[4,0]	])

b = np.array([	[1,2,3],
				[2,3,4],
				[3,4,5]	]	)

def calculate(a):
	d = np.linalg.det(a)
	eig, eig_vecs = np.linalg.eig(a)
	U, sig, Vt = np.linalg.svd(a)

	print("Determinant:", d)
	print("Eigen Values:", eig)
	print("Eigen Vectors:", eig_vecs)
	print("U-SVD:", U)
	print("Sigma-SVD", sig)
	print("Vt-SVD", Vt)

calculate(a)
