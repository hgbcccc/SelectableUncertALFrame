import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering
import logging
from scipy.spatial.distance import cdist



class SpectralFeatureExtractor:
    """光谱特征提取器"""
    
    def __init__(self, 
                 band_config: Optional[Dict[str, int]] = None,
                 index_weights: Optional[Dict[str, float]] = None,
                 selected_indices: Optional[List[str]] = None):
        """
        初始化光谱特征提取器
        
        Args:
            band_config: 波段配置字典，键为波段名称，值为波段索引
            index_weights: 指数权重字典，键为指数名称，值为权重
            selected_indices: 选定使用的指数列表
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 默认波段配置
        self.band_config = band_config or {
            'blue': 0,    # 蓝光波段索引
            'green': 1,   # 绿光波段索引
            'red': 2,     # 红光波段索引
            'nir': 3,     # 近红外波段索引
            'swir1': 4,   # 短波红外1波段索引
            'swir2': 5    # 短波红外2波段索引
        }
        
        # 必需的最小波段数
        self.min_bands = 4  # RGB + NIR
        
        # 支持的光谱指数列表
        self.supported_indices = {
            # 植被指数
            'ndvi': self._compute_ndvi,     # 归一化植被指数
            'evi': self._compute_evi,       # 增强型植被指数
            'savi': self._compute_savi,     # 土壤调节植被指数
            'msavi': self._compute_msavi,   # 修正的土壤调节植被指数
            'gndvi': self._compute_gndvi,   # 绿波段归一化植被指数
            'rvi': self._compute_rvi,       # 比值植被指数
            'dvi': self._compute_dvi,       # 差值植被指数
            
            # 水分指数
            'ndwi': self._compute_ndwi,     # 归一化水体指数
            'mndwi': self._compute_mndwi,   # 修正归一化水体指数
            
            # 叶绿素相关指数
            'mcari': self._compute_mcari,   # 修正的叶绿素吸收反射指数
            'tcari': self._compute_tcari,   # 转换叶绿素吸收反射指数
            'cri': self._compute_cri,       # 类胡萝卜素反射指数
            
            # 物候相关指数
            'ndsvi': self._compute_ndsvi,   # 归一化土壤植被指数
            'sipi': self._compute_sipi,     # 结构非敏感色素指数
            'pri': self._compute_pri,       # 光化学反射指数
        }
        
        # 设置默认的指数权重
        default_weights = {
            # 植被指数 (反映植被活力和生物量)
            'ndvi': 1.0,    # 基础植被指数，最重要
            'evi': 0.9,     # 改善了NDVI在高生物量区域的饱和问题
            
            # 水分指数 (反映植被水分状态)
            'ndwi': 0.8,    # 反映植被含水量
            'mndwi': 0.7,   # 使用SWIR波段，对水体更敏感
            
            # 叶绿素相关指数 (反映叶绿素含量)
            'mcari': 0.8,   # 对叶绿素含量变化敏感
            'tcari': 0.7,   # MCARI的改进版本
            'cri': 0.6,     # 类胡萝卜素指数
            
            # 结构敏感指数 (减少土壤背景影响)
            'savi': 0.8,    # 考虑土壤背景影响
            'msavi': 0.7,   # SAVI的改进版本
            
            # 物候相关指数 (反映生长阶段)
            'ndsvi': 0.7,   # 土壤植被指数
            'sipi': 0.6,    # 结构非敏感色素指数
            'pri': 0.6,     # 光化学反射指数
            
            # 其他补充指数
            'gndvi': 0.7,   # 绿波段归一化植被指数
            'rvi': 0.6,     # 比值植被指数
            'dvi': 0.5      # 差值植被指数
        }
        
        # 设置默认的选定指数
        default_selected = [
            # 基础组合
            'ndvi',    # 基础植被指数
            'ndwi',    # 水分指数
            'mcari',   # 叶绿素指数
            'savi',    # 土壤调节指数
            'ndsvi',   # 物候指数
            
            # 补充组合
            'evi',     # 增强植被指数
            'sipi',    # 结构非敏感色素指数
            'cri'      # 类胡萝卜素指数
        ]
        # 使用传入的参数或默认值
        self.index_weights = index_weights if index_weights is not None else default_weights
        self.selected_indices = selected_indices if selected_indices is not None else default_selected


    def _validate_input(self, img: np.ndarray) -> bool:
        """验证输入数据是否符合要求
        
        Args:
            img: 输入图像数组
            
        Returns:
            bool: 验证是否通过
            
        Raises:
            ValueError: 当输入数据不符合要求时
        """
        if not isinstance(img, np.ndarray):
            raise ValueError(f"输入必须是numpy数组，而不是{type(img)}")
            
        if len(img.shape) != 3:
            raise ValueError(f"输入必须是3维数组 (H,W,C)，而不是{len(img.shape)}维")
            
        if img.shape[2] < self.min_bands:
            raise ValueError(f"输入必须至少包含{self.min_bands}个波段（RGB+NIR），"
                           f"当前波段数：{img.shape[2]}")
            
        # 检查波段配置是否有效
        max_band_idx = max(self.band_config.values())
        if max_band_idx >= img.shape[2]:
            raise ValueError(f"波段配置索引超出范围，最大索引为{max_band_idx}，"
                           f"但图像只有{img.shape[2]}个波段")
            
        return True

    
    def _compute_ndvi(self, img: np.ndarray) -> np.ndarray:
        """计算归一化植被指数 (Normalized Difference Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        return (nir - red) / (nir + red + 1e-10)

    def _compute_evi(self, img: np.ndarray) -> np.ndarray:
        """计算增强型植被指数 (Enhanced Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        blue = img[..., self.band_config['blue']]
        return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)

    def _compute_savi(self, img: np.ndarray) -> np.ndarray:
        """计算土壤调节植被指数 (Soil Adjusted Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        L = 0.5  # 土壤亮度校正因子
        return ((nir - red) / (nir + red + L + 1e-10)) * (1 + L)

    def _compute_msavi(self, img: np.ndarray) -> np.ndarray:
        """计算修正的土壤调节植被指数 (Modified Soil Adjusted Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        return (2 * nir + 1 - np.sqrt((2 * nir + 1)**2 - 8 * (nir - red))) / 2

    def _compute_gndvi(self, img: np.ndarray) -> np.ndarray:
        """计算绿波段归一化植被指数 (Green Normalized Difference Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        green = img[..., self.band_config['green']]
        return (nir - green) / (nir + green + 1e-10)

    def _compute_rvi(self, img: np.ndarray) -> np.ndarray:
        """计算比值植被指数 (Ratio Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        return nir / (red + 1e-10)

    def _compute_dvi(self, img: np.ndarray) -> np.ndarray:
        """计算差值植被指数 (Difference Vegetation Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        return nir - red

    def _compute_ndwi(self, img: np.ndarray) -> np.ndarray:
        """计算归一化水体指数 (Normalized Difference Water Index)"""
        green = img[..., self.band_config['green']]
        nir = img[..., self.band_config['nir']]
        return (green - nir) / (green + nir + 1e-10)

    def _compute_mndwi(self, img: np.ndarray) -> np.ndarray:
        """计算修正归一化水体指数 (Modified Normalized Difference Water Index)"""
        green = img[..., self.band_config['green']]
        swir1 = img[..., self.band_config['swir1']]
        return (green - swir1) / (green + swir1 + 1e-10)

    def _compute_mcari(self, img: np.ndarray) -> np.ndarray:
        """计算修正的叶绿素吸收反射指数 (Modified Chlorophyll Absorption Ratio Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        green = img[..., self.band_config['green']]
        return ((nir - red) - 0.2 * (nir - green)) * (nir / red)

    def _compute_tcari(self, img: np.ndarray) -> np.ndarray:
        """计算转换叶绿素吸收反射指数 (Transformed Chlorophyll Absorption Ratio Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        green = img[..., self.band_config['green']]
        return 3 * ((nir - red) - 0.2 * (nir - green) * (nir / red))

    def _compute_cri(self, img: np.ndarray) -> np.ndarray:
        """计算类胡萝卜素反射指数 (Carotenoid Reflection Index)"""
        blue = img[..., self.band_config['blue']]
        green = img[..., self.band_config['green']]
        return (1 / blue) - (1 / green)

    def _compute_ndsvi(self, img: np.ndarray) -> np.ndarray:
        """计算归一化土壤植被指数 (Normalized Difference Soil Vegetation Index)"""
        swir1 = img[..., self.band_config['swir1']]
        red = img[..., self.band_config['red']]
        return (swir1 - red) / (swir1 + red + 1e-10)

    def _compute_sipi(self, img: np.ndarray) -> np.ndarray:
        """计算结构非敏感色素指数 (Structure Insensitive Pigment Index)"""
        nir = img[..., self.band_config['nir']]
        red = img[..., self.band_config['red']]
        blue = img[..., self.band_config['blue']]
        return (nir - blue) / (nir - red + 1e-10)

    def _compute_pri(self, img: np.ndarray) -> np.ndarray:
        """计算光化学反射指数 (Photochemical Reflectance Index)"""
        blue = img[..., self.band_config['blue']]
        green = img[..., self.band_config['green']]
        return (blue - green) / (blue + green + 1e-10)

    def compute_statistical_features(self, index_array: np.ndarray) -> Dict[str, float]:
            """计算光谱指数的统计特征"""
            flat_array = index_array.flatten()
            return {
                'mean': float(np.mean(flat_array)),
                'std': float(np.std(flat_array)),
                'skewness': float(stats.skew(flat_array)),
                'kurtosis': float(stats.kurtosis(flat_array)),
                'min': float(np.min(flat_array)),
                'max': float(np.max(flat_array)),
                'range': float(np.ptp(flat_array)),
                'variance': float(np.var(flat_array)),
                'median': float(np.median(flat_array)),
                'p25': float(np.percentile(flat_array, 25)),
                'p75': float(np.percentile(flat_array, 75)),
                'iqr': float(np.percentile(flat_array, 75) - np.percentile(flat_array, 25))
            }

    def calculate_spectral_indices(self, 
                                 img: np.ndarray, 
                                 indices: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """计算多种光谱指数"""
        self._validate_input(img)
        
        if indices is None:
            indices = list(self.supported_indices.keys())
        else:
            unsupported = set(indices) - set(self.supported_indices.keys())
            if unsupported:
                raise ValueError(f"不支持的光谱指数：{unsupported}")
        
        results = {}
        for index_name in indices:
            try:
                results[index_name] = self.supported_indices[index_name](img)
            except Exception as e:
                self.logger.error(f"计算{index_name}时出错: {str(e)}")
                continue
                
        return results


    def extract_features(self, 
                        img: np.ndarray, 
                        indices: Optional[List[str]] = None,
                        normalize: bool = True) -> Dict[str, np.ndarray]:
        """提取光谱特征"""
        # 使用选定的指数列表
        indices = indices or self.selected_indices
        
        # 计算光谱指数
        spectral_indices = self.calculate_spectral_indices(img, indices)
        
        # 提取每个指数的统计特征并应用权重
        features = {}
        for index_name, index_values in spectral_indices.items():
            # 获取权重，如果未指定则使用默认值1.0
            weight = self.index_weights.get(index_name, 1.0)
            
            # 计算统计特征
            stats = self.compute_statistical_features(index_values)
            
            # 应用权重
            weighted_stats = np.array(list(stats.values())) * weight
            features[index_name] = weighted_stats
            
        # 特征归一化
        if normalize:
            scaler = StandardScaler()
            features_array = np.vstack(list(features.values()))
            normalized_features = scaler.fit_transform(features_array)
            features = {
                name: normalized_features[i] 
                for i, name in enumerate(features.keys())
            }
            
        return features
    
    
    def get_index_weights(self) -> Dict[str, float]:
        """获取当前使用的指数权重"""
        return {
            index: self.index_weights.get(index, 1.0)
            for index in self.selected_indices
        }

    def set_index_weights(self, weights: Dict[str, float]) -> None:
        """设置指数权重"""
        self.index_weights.update(weights)
        self.logger.info(f"更新指数权重: {weights}")

    def set_selected_indices(self, indices: List[str]) -> None:
        """设置要使用的指数组合"""
        unsupported = set(indices) - set(self.supported_indices.keys())
        if unsupported:
            raise ValueError(f"不支持的光谱指数：{unsupported}")
        
        self.selected_indices = indices
        self.logger.info(f"更新选定的指数组合: {indices}")

    def analyze_index_correlation(self, features_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """分析不同光谱指数之间的相关性"""
        indices = list(features_dict.keys())
        n_indices = len(indices)
        corr_matrix = np.zeros((n_indices, n_indices))
        
        for i, idx1 in enumerate(indices):
            for j, idx2 in enumerate(indices):
                corr = np.corrcoef(
                    features_dict[idx1].flatten(),
                    features_dict[idx2].flatten()
                )[0, 1]
                corr_matrix[i, j] = corr
                
        return corr_matrix, indices

    def compute_complementarity(self, features_dict: Dict[str, np.ndarray]) -> float:
        """计算指数间的互补性"""
        corr_matrix, _ = self.analyze_index_correlation(features_dict)
        # 使用相关系数的倒数作为互补性度量
        complementarity = 1 / (np.mean(np.abs(corr_matrix)) + 1e-10)
        return float(complementarity)



class SpectralSamplingStrategy:
    """多光谱采样策略类"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 支持的距离度量方法
        self.distance_metrics = {
            'euclidean': self._euclidean_distance,
            'cosine': self._cosine_distance,
            'mahalanobis': self._mahalanobis_distance,
            'mmd': self._maximum_mean_discrepancy
        }
        
        # 支持的采样策略
        self.strategies = {
            'max_min_distance': self.sample_by_max_min_distance,
            'cluster': self.sample_by_clustering,
            'coreset': self.sample_by_coreset,
            'diversity': self.sample_by_diversity
        }
        self._distance_cache = {}
        self._similarity_cache = {}

    
    
    def _get_distance_matrix(self, 
                           features: np.ndarray, 
                           distance_metric: str) -> np.ndarray:
        """获取或计算距离矩阵（带缓存）"""
        key = (features.tobytes(), distance_metric)
        if key not in self._distance_cache:
            distance_func = self.distance_metrics[distance_metric]
            self._distance_cache[key] = distance_func(features, features)
        return self._distance_cache[key]
    
    def evaluate_sampling(self,
                         features: np.ndarray,
                         selected_indices: np.ndarray,
                         distance_metric: str = 'euclidean') -> Dict[str, float]:
        """评估采样结果的质量
        
        Args:
            features: 特征矩阵
            selected_indices: 选中的样本索引
            distance_metric: 距离度量方法
            
        Returns:
            Dict[str, float]: 评估指标
        """
        metrics = self.compute_spectral_metrics(
            features, selected_indices, distance_metric
        )
        
        # 添加额外的评估指标
        selected_features = features[selected_indices]
        
        # 计算样本分布均匀性
        distances = self._get_distance_matrix(selected_features, distance_metric)
        uniformity = np.std(distances) / np.mean(distances)
        metrics['uniformity'] = float(uniformity)
        
        # 计算覆盖率
        all_distances = self._get_distance_matrix(features, distance_metric)
        coverage_radius = np.percentile(all_distances, 90)
        covered = (all_distances <= coverage_radius).any(axis=1)
        coverage_rate = covered.mean()
        metrics['coverage_rate'] = float(coverage_rate)
        
        return metrics
    
    def _validate_inputs(self, 
                        features: np.ndarray, 
                        n_samples: int,
                        distance_metric: str) -> None:
        """验证输入参数
        
        Args:
            features: 特征矩阵
            n_samples: 采样数量
            distance_metric: 距离度量方法
        """
        if not isinstance(features, np.ndarray):
            raise TypeError("features必须是numpy数组")
        
        if features.ndim != 2:
            raise ValueError("features必须是二维数组")
            
        if n_samples <= 0:
            raise ValueError("n_samples必须大于0")
            
        if n_samples > len(features):
            raise ValueError("n_samples不能大于样本总数")
            
        if distance_metric not in self.distance_metrics:
            raise ValueError(f"不支持的距离度量方法：{distance_metric}")
        
    def _euclidean_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """计算欧氏距离"""
        return euclidean_distances(X, Y)

    def _cosine_distance(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """计算余弦距离"""
        X_norm = X / np.linalg.norm(X, axis=1)[:, np.newaxis]
        Y_norm = Y / np.linalg.norm(Y, axis=1)[:, np.newaxis]
        return 1 - np.dot(X_norm, Y_norm.T)

    def _mahalanobis_distance(self, 
                                X: np.ndarray, 
                                Y: np.ndarray,
                                n_jobs: int = -1) -> np.ndarray:
            """并行计算马氏距离"""
            from joblib import Parallel, delayed
            
            features = np.vstack([X, Y])
            covariance = np.cov(features.T)
            inv_covariance = np.linalg.pinv(covariance)
            
            def compute_distance(i, j):
                diff = X[i] - Y[j]
                return np.sqrt(diff.dot(inv_covariance).dot(diff))
            
            distances = Parallel(n_jobs=n_jobs)(
                delayed(compute_distance)(i, j)
                for i in range(len(X))
                for j in range(len(Y))
            )
            return np.array(distances).reshape(len(X), len(Y))
    
    def _maximum_mean_discrepancy(self, X: np.ndarray, Y: np.ndarray, gamma: float = 1.0) -> float:
        """计算最大均值差异(MMD)
        
        Args:
            X: 第一个样本集的特征矩阵
            Y: 第二个样本集的特征矩阵
            gamma: 高斯核参数
            
        Returns:
            float: MMD距离
        """
        def gaussian_kernel(x: np.ndarray, y: np.ndarray) -> float:
            return np.exp(-gamma * np.linalg.norm(x - y) ** 2)
        
        XX = np.mean([gaussian_kernel(x1, x2) for x1 in X for x2 in X])
        YY = np.mean([gaussian_kernel(y1, y2) for y1 in Y for y2 in Y])
        XY = np.mean([gaussian_kernel(x, y) for x in X for y in Y])
        
        return XX + YY - 2 * XY
    
    
    def sample_by_max_min_distance(self, 
                                features: np.ndarray, 
                                n_samples: int,
                                distance_metric: str = 'euclidean',
                                initial_idx: Optional[int] = None) -> np.ndarray:
        """基于最大最小距离的采样策略
        
        Args:
            features: 特征矩阵 (n_samples, n_features)
            n_samples: 要选择的样本数量
            distance_metric: 距离度量方法
            initial_idx: 初始样本索引，如果为None则随机选择
            
        Returns:
            np.ndarray: 选中样本的索引
        """
        # 输入验证
        self._validate_inputs(features, n_samples, distance_metric)
        
        if len(features) <= n_samples:
            return np.arange(len(features))
            
        # 初始化
        if initial_idx is None:
            initial_idx = np.random.randint(len(features))
        selected = [initial_idx]
        
        # 获取距离矩阵
        distances = self._get_distance_matrix(features, distance_metric)
        
        # 迭代选择样本
        while len(selected) < n_samples:
            if len(selected) % 10 == 0:
                self.logger.info(f"已选择 {len(selected)}/{n_samples} 个样本")
                
            # 计算到已选样本的最小距离
            min_distances = distances[:, selected].min(axis=1)
            
            # 选择具有最大最小距离的样本
            remaining_mask = ~np.isin(np.arange(len(features)), selected)
            remaining_dist = min_distances[remaining_mask]
            max_idx = np.argmax(remaining_dist)
            
            # 获取原始索引
            original_idx = np.arange(len(features))[remaining_mask][max_idx]
            selected.append(original_idx)
            
        return np.array(selected)

    def sample_by_clustering(self,
                            features: np.ndarray,
                            n_samples: int,
                            distance_metric: str = 'euclidean',
                            n_clusters: Optional[int] = None) -> np.ndarray:
        """基于聚类的采样策略"""
        # 添加输入验证
        self._validate_inputs(features, n_samples, distance_metric)
        
        if n_clusters is None:
            n_clusters = min(n_samples, len(features))
            
        # 根据距离度量选择聚类方法
        if distance_metric == 'euclidean':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(features)
        else:
            # 使用缓存的距离矩阵
            distances = self._get_distance_matrix(features, distance_metric)
            similarity = np.exp(-distances)
            clusterer = SpectralClustering(
                n_clusters=n_clusters,
                affinity='precomputed',
                random_state=42
            )
            cluster_labels = clusterer.fit_predict(similarity)
        
        selected_indices = []
        for cluster_idx in range(n_clusters):
            cluster_mask = cluster_labels == cluster_idx
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # 选择簇中心最近的样本
            center = features[cluster_mask].mean(axis=0)
            distances = self._get_distance_matrix(
                features[cluster_mask],
                distance_metric
            )
            n_select = max(1, int(n_samples * len(cluster_indices) / len(features)))
            selected = cluster_indices[np.argsort(distances.flatten())[:n_select]]
            selected_indices.extend(selected)
            
        return np.array(selected_indices[:n_samples])
            
        

    def sample_by_coreset(self,
                        features: np.ndarray,
                        n_samples: int,
                        distance_metric: str = 'euclidean',
                        greedy: bool = True) -> np.ndarray:
        """基于核心集的采样策略
        
        Args:
            features: 特征矩阵
            n_samples: 要选择的样本数量
            distance_metric: 距离度量方法
            greedy: 是否使用贪心策略
            
        Returns:
            np.ndarray: 选中样本的索引
        """
        # 输入验证
        self._validate_inputs(features, n_samples, distance_metric)
        
        # 如果使用贪心策略，直接使用最大最小距离采样
        if greedy:
            return self.sample_by_max_min_distance(
                features=features,
                n_samples=n_samples,
                distance_metric=distance_metric
            )
        
        # 获取距离矩阵
        distances = self._get_distance_matrix(features, distance_metric)
        
        # 初始化选择
        selected_indices = []
        current_idx = np.random.randint(len(features))
        selected_indices.append(current_idx)
        
        # 迭代选择样本
        while len(selected_indices) < n_samples:
            if len(selected_indices) % 10 == 0:
                self.logger.info(f"已选择 {len(selected_indices)}/{n_samples} 个样本")
                
            # 计算到已选样本的最小距离
            min_distances = distances[:, selected_indices].min(axis=1)
            
            # 计算选择概率（距离的平方）
            probs = min_distances ** 2
            probs[selected_indices] = 0  # 已选样本概率设为0
            probs /= probs.sum()  # 归一化概率
            
            # 按概率选择下一个样本
            next_idx = np.random.choice(len(features), p=probs)
            selected_indices.append(next_idx)
        
        return np.array(selected_indices)

    def sample_by_diversity(self,
                        features: np.ndarray,
                        n_samples: int,
                        distance_metric: str = 'euclidean',
                        temperature: float = 1.0) -> np.ndarray:
        """基于多样性的采样策略
        
        Args:
            features: 特征矩阵
            n_samples: 要选择的样本数量
            distance_metric: 距离度量方法
            temperature: 温度参数，控制多样性程度
            
        Returns:
            np.ndarray: 选中样本的索引
        """
        # 输入验证
        self._validate_inputs(features, n_samples, distance_metric)
        
        # 获取距离矩阵
        distances = self._get_distance_matrix(features, distance_metric)
        
        # 计算相似度矩阵
        similarity = np.exp(-distances / temperature)
        
        # 初始化选择
        selected_indices = []
        current_idx = np.random.randint(len(features))
        selected_indices.append(current_idx)
        
        # 迭代选择样本
        while len(selected_indices) < n_samples:
            if len(selected_indices) % 10 == 0:
                self.logger.info(f"已选择 {len(selected_indices)}/{n_samples} 个样本")
                
            # 计算候选样本的多样性分数
            candidate_scores = np.zeros(len(features))
            
            # 计算每个未选择样本的多样性分数
            for i in range(len(features)):
                if i in selected_indices:
                    continue
                # 计算与已选样本的平均相似度
                sim_to_selected = similarity[i, selected_indices].mean()
                candidate_scores[i] = -sim_to_selected  # 负相似度作为多样性分数
            
            # 选择具有最高多样性分数的样本
            remaining_mask = ~np.isin(np.arange(len(features)), selected_indices)
            remaining_scores = candidate_scores[remaining_mask]
            max_idx = np.argmax(remaining_scores)
            original_idx = np.arange(len(features))[remaining_mask][max_idx]
            selected_indices.append(original_idx)
        
        return np.array(selected_indices)

    def sample(self,
            features: np.ndarray,
            n_samples: int,
            strategy: str = 'max_min_distance',
            distance_metric: str = 'euclidean',
            **kwargs) -> np.ndarray:
        """统一的采样接口
        
        Args:
            features: 特征矩阵
            n_samples: 要选择的样本数量
            strategy: 采样策略
            distance_metric: 距离度量方法
            **kwargs: 其他参数
            
        Returns:
            np.ndarray: 选中样本的索引
        """
        if strategy not in self.strategies:
            raise ValueError(f"不支持的采样策略：{strategy}")
        if distance_metric not in self.distance_metrics:
            raise ValueError(f"不支持的距离度量方法：{distance_metric}")
            
        sampling_func = self.strategies[strategy]
        return sampling_func(
            features=features,
            n_samples=n_samples,
            distance_metric=distance_metric,
            **kwargs
        )

    def compute_spectral_metrics(self,
                               features: np.ndarray,
                               selected_indices: np.ndarray,
                               distance_metric: str = 'euclidean') -> Dict[str, float]:
        """计算光谱采样的评估指标"""
        distance_func = self.distance_metrics[distance_metric]
        metrics = {}
        
        # 计算覆盖度
        all_distances = distance_func(features, features[selected_indices])
        coverage = np.min(all_distances, axis=1).mean()
        metrics['coverage'] = float(coverage)
        
        # 计算多样性
        selected_features = features[selected_indices]
        diversity = distance_func(selected_features, selected_features).mean()
        metrics['diversity'] = float(diversity)
        
        # 计算代表性
        selected_center = selected_features.mean(axis=0)
        total_center = features.mean(axis=0)
        representativeness = np.linalg.norm(selected_center - total_center)
        metrics['representativeness'] = float(representativeness)
        
        return metrics