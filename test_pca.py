import unittest
import pca
import numpy as np
from sklearn.datasets import load_iris
from statistics import variance
from sklearn.decomposition import PCA

# irisデータを使って自作のPCAコードをsklearnと比較してテストする


class TestPca(unittest.TestCase):
  iris = load_iris()
  pca = pca.my_PCA(iris.data)
  test_pca = PCA()
  test_pca.fit(iris.data)
  # 分散のテスト
  def test_calc_variance(self):
    variances = [variance(self.pca.data[:,i]) for i in range(self.pca.dimension)]
    test_variances = np.sum((self.pca.data**2)/(self.pca.n-1), axis=0)
    # print((variances - test_variances))
    self.assertEqual(True, ((variances - test_variances) < 0.0000001).all())

  # 分散共分散行列のテスト
  def test_get_Covariance_Matrix(self):
      cvm = self.pca.get_Covariance_Matrix()
      test_cvm = np.cov(self.pca.data, rowvar=0, bias=0)
      # print((cvm - test_cvm))
      self.assertEqual(True, ((cvm - test_cvm) < 0.0000001).all())

  # 主成分ベクトルのテスト
  def test_get_components(self):
      cvm = self.pca.get_Covariance_Matrix()
      components = self.pca.get_components(cvm)
      test_components = self.test_pca.components_
      self.assertEqual(True, ((abs(components) - abs(test_components)) < 0.0000001).all())

if __name__ == '__main__':
    unittest.main()
