import torch.nn as nn
from banzhaf import BanzhafModule
from utils import CrossEn,KL
import torch.nn.functional as F
import torch



def get_similar(view0_feat, view1_feat):
    # 归一化特征
    view0_feat = view0_feat / view0_feat.norm(dim=-1, keepdim=True)
    view1_feat = view1_feat / view1_feat.norm(dim=-1, keepdim=True)
    # 计算相似度矩阵
    retrieve_logits = torch.einsum('ad,bd->ab', [view0_feat, view1_feat])  # [N, N]
    _retrieve_logits = (retrieve_logits + retrieve_logits.T) / 2
    return _retrieve_logits, _retrieve_logits.T, retrieve_logits


class RepresentationReconstruction(nn.Module):
    """
    根据论文图5的“表征重构模块”的核心思想，重构两个多视图潜在表示。
    它包含交叉注意力机制和带权重的残差连接。
    输入表示的维度为 (N, D)。
    """
    def __init__(self, feature_dim: int):
        """
        初始化特征重构模块。

        Args:
            feature_dim (int): 潜在表示的维度 D。
        """
        super().__init__()
        self.feature_dim = feature_dim

        # MLP for dynamic weighting factor alpha for View1
        # 输入形状 (N1, D)，输出形状 (N1, 1)。
        self.alpha = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid() # 确保输出在 [0, 1] 之间
        )

        # MLP for dynamic weighting factor alpha for View2
        # 输入形状 (N2, D)，输出形状 (N2, 1)。
        self.beta = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid() # 确保输出在 [0, 1] 之间
        )

    def forward(
        self,
        view1_latent,         # N1, D (例如，原始文本特征)
        view2_latent,         # N2, D (例如，原始视频帧特征)
    ):
        """
        执行多视图表示的重构。

        Args:
            view1_latent_s (torch.Tensor): 第一个视图的单模态潜在特征，形状 (N1, D)。
            view2_latent_s (torch.Tensor): 第二个视图的单模态潜在特征，形状 (N2, D)。
            view1_global_context (Optional[torch.Tensor]): (可选) 第一个视图的全局上下文表示，形状 (D,)。
                                                        如果未提供，将对 view1_latent_s 进行平均池化。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - reconstructed_view1_latent (torch.Tensor): 重构后的第一个视图特征，形状 (N1, D)。
                - reconstructed_view2_latent (torch.Tensor): 重构后的第二个视图特征，形状 (N2, D)。
        """

        # --- 重构 View2 (由 View1 的上下文驱动) ---

        # 1. 获取 View1 的全局上下文 (作为 Query)
        query_view1 = view1_latent.mean(dim=0)

        # 2. 计算 View2 的跨模态表示 (View2_c)
        # 交叉注意力：Query (View1_ctx) 对 Keys/Values (View2_latent_s)
        attention_view2 = torch.matmul(view2_latent, query_view1)
        attention_weights_view2 = F.softmax(attention_view2, dim=0).unsqueeze(-1) # (N2, 1)

        # View2 的跨模态表示：View2_latent_s 的加权平均
        view2_latent_c = view2_latent * attention_weights_view2 # (N2, D)

        # 3. 计算融合权重 alpha (这里对应论文中的 gamma)
        diff_view2 = view2_latent- view2_latent_c # (N2, D)
        alpha_view2 = self.alpha(diff_view2) # (N2, 1)

        # 4. 融合 View2 的单模态和跨模态表示
        reconstructed_view2 = alpha_view2 * view2_latent + (1 - alpha_view2) * view2_latent_c # (N2, D)


        # --- 重构 View1 (由 View2 的上下文驱动) ---

        # 1. 获取 View2 的全局上下文 (作为 Query)
        query_view2 = view2_latent.mean(dim=0) # (D,)

        # 2. 计算 View1 的跨模态表示 (View1_c)
        # 交叉注意力：Query (View2_ctx) 对 Keys/Values (View1_latent_s)
        attention_view1 = torch.matmul(view1_latent, query_view2)
        attention_view1 = F.softmax(attention_view1, dim=0).unsqueeze(-1) # (N1, 1)

        # View1 的跨模态表示：View1_latent_s 的加权平均
        view1_latent_c = view1_latent * attention_view1 # (N1, D)

        # 3. 计算融合权重 alpha (这里对应论文中的 delta)
        diff_view1 = view1_latent - view1_latent_c # (N1, D)
        beta_view1 = self.beta(diff_view1) # (N1, 1)

        # 4. 融合 View1 的单模态和跨模态表示
        reconstructed_view1 = beta_view1 * view1_latent + (1 - beta_view1) * view1_latent_c # (N1, D)

        return reconstructed_view1, reconstructed_view2

class Autoencoder(nn.Module):
    """AutoEncoder module that projects features to latent space."""

    def __init__(self,
                 encoder_dim,
                 activation='relu',
                 batchnorm=True):

        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1
        self._activation = activation
        self._batchnorm = batchnorm

        encoder_layers = []
        for i in range(self._dim):
            encoder_layers.append(
                nn.Linear(encoder_dim[i], encoder_dim[i + 1]))
            if i < self._dim - 1:
                if self._batchnorm:
                    encoder_layers.append(nn.BatchNorm1d(encoder_dim[i + 1]))
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        # encoder_layers.append(nn.Softmax(dim=1))
        self._encoder = nn.Sequential(*encoder_layers)

        decoder_dim = [i for i in reversed(encoder_dim)]
        decoder_layers = []
        for i in range(self._dim):
            decoder_layers.append(
                nn.Linear(decoder_dim[i], decoder_dim[i + 1]))
            if self._batchnorm:
                decoder_layers.append(nn.BatchNorm1d(decoder_dim[i + 1]))
            if self._activation == 'sigmoid':
                decoder_layers.append(nn.Sigmoid())
            elif self._activation == 'leakyrelu':
                encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
            elif self._activation == 'tanh':
                decoder_layers.append(nn.Tanh())
            elif self._activation == 'relu':
                decoder_layers.append(nn.ReLU())
            else:
                raise ValueError('Unknown activation type %s' % self._activation)
        decoder_layers = decoder_layers[:-1]
        self._decoder = nn.Sequential(*decoder_layers)

    def encoder(self, x):
        latent = self._encoder(x)
        return latent

    def decoder(self, latent):
        x_hat = self._decoder(latent)
        return x_hat

    def forward(self, x):
        latent = self.encoder(x)
        x_hat = self.decoder(latent)
        return x_hat, latent

class Encoder(nn.Module):
    def __init__(self, x_dim,z_dim):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,z_dim)
        )

    def forward(self,x):
        return F.normalize(self.encoder(x))

class Decoder(nn.Module):
    def __init__(self, x_dim,z_dim):
        super(Decoder,self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(z_dim,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,x_dim)
        )
    def forward(self,z):
        return self.decoder(z)

class MLP(nn.Module):
    def __init__(self, code_dim, layers):
        super(MLP, self).__init__()
        self.code_dim = code_dim
        self.layers = layers
        self.hidden = nn.ModuleList()

        for k in range(layers):
            linear_layer = nn.Linear(code_dim, code_dim, bias=False)
            self.hidden.append(linear_layer)

    def forward(self, z):
        for l in self.hidden:
            z = l(z)
        return z


class Network(nn.Module):
    def __init__(self,args, dim_list,device):
        super(Network,self).__init__()
        self.args = args
        self._latent_dim = args.n_z

        self.class_dim =args.num_cluster

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.encoders.append(Encoder(dim_list[0], self._latent_dim))
        self.encoders.append(Encoder(dim_list[1], self._latent_dim))
        self.decoders.append(Decoder(dim_list[0], self._latent_dim))
        self.decoders.append(Decoder(dim_list[1], self._latent_dim))
        self.kl = KL()
        self.banzhafmodel = BanzhafModule(64).to(device)
        self.banzhafmodel0 = BanzhafModule(64).to(device)
        self.banzhafteacher = BanzhafModule(64).to(device)
        self.best_scores_kmeans = [0, 0, 0, 0, 0, 0, 0, 0]
        self.best_epoch = 0
        self.cluster = nn.Sequential(
            nn.Linear(self._latent_dim, self._latent_dim),
            nn.Linear(self._latent_dim, self.class_dim),
            nn.Softmax(dim=1)
        )
        # self.mlps = nn.ModuleList()
        # self.mlps.append(MLP(self._latent_dim,2))
        # self.mlps.append(MLP(self._latent_dim,2))
        self.reconstruct_feature = RepresentationReconstruction(self._latent_dim)


    def forward(self,data0,data1):
        z0 = self.encoders[0](data0)
        z1 = self.encoders[1](data1)
        # r_z0, r_z1 = self.reconstruct_feature(z0, z1)
        re_data0 = self.decoders[0](z0)
        re_data1 = self.decoders[1](z1)
        M_t2v_logits, M_v2t_logits, logits = get_similar(z0, z1)
        banzhaf = self.banzhafmodel(logits.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            teacher = self.banzhafteacher(logits.unsqueeze(1).clone().detach()).squeeze(1).detach()
        # cluster0 = self.cluster(z0)
        # cluster1 = self.cluster(z1)
        # C_t2v_logits, C_v2t_logits, Clogits = get_similar(cluster0, cluster1)
        # banzhaf1 = self.banzhafmodel0(Clogits.unsqueeze(1)).squeeze(1)
        # with torch.no_grad():
        #     teacher1 = self.banzhafteacher(Clogits.unsqueeze(1).clone().detach()).squeeze(1).detach()

        return re_data0, re_data1,z0,z1,M_t2v_logits, M_v2t_logits, logits,banzhaf,teacher
