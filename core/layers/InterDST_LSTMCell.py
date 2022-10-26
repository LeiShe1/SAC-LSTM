__author__ = 'chuyao'

import torch
import torch.nn as nn

class InterDST_LSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm,r,method=0,num_layer=1):
        super(InterDST_LSTMCell, self).__init__()

        self.method = method
        self.num_layer = num_layer
        self.r = r
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.layer_norm = nn.LayerNorm([num_hidden,width,width])
        self.c_norm = nn.LayerNorm([num_hidden, width, width])
        self.s_norm = nn.LayerNorm([num_hidden, width, width])



        self.c_attn_ = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width]),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.9)
        )
        self.s_attn_ = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width]),
            nn.ReLU(),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([num_hidden, width, width]),
            # nn.ReLU(),
            # nn.Dropout2d(p=0.9)
        )
        self.attn_ = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=1, stride=1, padding=0),
            # nn.LayerNorm([num_hidden, width, width])
            # nn.Dropout2d(p=0.9)
        )
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 7, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, width, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, width, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

        self.conv_x_h = []
        self.conv_x_x = []
        self.conv_h_x = []
        self.conv_h_h = []

        for i in range(self.r):
            self.conv_x_h.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([num_hidden, width, width])
                )
            )
            self.conv_x_x.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, in_channel, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([in_channel, width, width])
                )
            )
            self.conv_h_x.append(
                nn.Sequential(
                    nn.Conv2d(num_hidden, in_channel, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([in_channel, width, width])
                )
            )
            self.conv_h_h.append(
                nn.Sequential(
                    nn.Conv2d(num_hidden, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
                    nn.LayerNorm([num_hidden, width, width])
                )
            )
        self.conv_x_h = nn.ModuleList(self.conv_x_h)
        self.conv_x_x = nn.ModuleList(self.conv_x_x)
        self.conv_h_x = nn.ModuleList(self.conv_h_x)
        self.conv_h_h = nn.ModuleList(self.conv_h_h)

        if self.method == 1:
            dim = width ** 2
            self.encoder_layer_1 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_1 = torch.nn.TransformerEncoder(
                self.encoder_layer_1,
                num_layers=4)
            self.encoder_layer_2 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_2 = torch.nn.TransformerEncoder(
                self.encoder_layer_2,
                num_layers=2)
            self.encoder_layer_3 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_3 = torch.nn.TransformerEncoder(
                self.encoder_layer_3,
                num_layers=2)

            self.conv1x1_1 = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=(1, 1), stride=(1, 1), padding=0)
            self.conv1x1_2 = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=(1, 1), stride=(1, 1), padding=0)
            self.conv1x1_3 = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=(1, 1), stride=(1, 1), padding=0)

        if self.method == 2:
            dim = width ** 2
            self.encoder_layer_1 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_1 = torch.nn.TransformerEncoder(
                self.encoder_layer_1,
                num_layers=2)
            self.encoder_layer_2 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_2 = torch.nn.TransformerEncoder(
                self.encoder_layer_2,
                num_layers=2)
            self.encoder_layer_3 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_3 = torch.nn.TransformerEncoder(
                self.encoder_layer_3,
                num_layers=2)

        if self.method == 3:
            dim = width ** 2
            self.encoder_layer_1 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_1 = torch.nn.TransformerEncoder(
                self.encoder_layer_1,
                num_layers=self.num_layer)
            self.encoder_layer_2 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_2 = torch.nn.TransformerEncoder(
                self.encoder_layer_2,
                num_layers=self.num_layer)
            self.encoder_layer_3 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_3 = torch.nn.TransformerEncoder(
                self.encoder_layer_3,
                num_layers=self.num_layer)
            self.encoder_layer_4 = torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=4,
                dim_feedforward=2048)
            self.encoder_model_4 = torch.nn.TransformerEncoder(
                self.encoder_layer_4,
                num_layers=self.num_layer)

    def _attn_channel(self,in_query,in_keys,in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch,num_channels,-1])
        key = in_keys.reshape([batch,-1,height*width]).permute((0, 2, 1))
        value = in_values.reshape([batch,-1,height*width]).permute((0, 2, 1))
        attn = torch.matmul(query,key)
        attn = torch.nn.Softmax(dim=2)(attn)
        attn = torch.matmul(attn,value.permute(0,2,1))
        attn = attn.reshape([batch,num_channels,width,height])

        return attn

    def _attn_spatial(self,in_query,in_keys,in_values):
        q_shape = in_query.shape
        k_shape = in_keys.shape
        batch = q_shape[0]
        num_channels = q_shape[1]
        width = q_shape[2]
        height = q_shape[3]
        length = k_shape[1]
        query = in_query.reshape([batch,num_channels,-1]).permute((0,2,1))
        key = in_keys.permute((0,1,3,4,2)).reshape([batch,-1,num_channels])
        value = in_values.permute((0, 1, 3, 4, 2)).reshape([batch, -1, num_channels])
        attn = torch.matmul(query,key.permute(0,2,1))
        attn = torch.nn.Softmax(dim=2)(attn)
        attn = torch.matmul(attn,value)
        attn = attn.reshape([batch,width,height,num_channels]).permute(0,3,1,2)

        return attn

    def attn_sum_fussion(self,c ,in_query,in_keys,in_values):
        spatial_attn = self.s_norm(c + self._attn_spatial(in_query, in_keys, in_values))
        channel_attn = self.c_norm(c + self._attn_channel(in_query, in_keys, in_values))

        s_attn = self.s_attn_(spatial_attn)
        c_attn = self.c_attn_(channel_attn)
        attn = s_attn + c_attn

        attn = self.attn_(attn)
        return attn

    def forward(self, x_t, h_t, c_t,c_historys, m_t):


        if self.method==3:
            b_h, c_h, w_h, h_h = h_t.size()
            h_tf = h_t.reshape(b_h, c_h, w_h * h_h)  # b w c*h
            b_c, c_c, w_c, h_c = c_t.size()
            c_tf = c_t.reshape(b_c, c_c, w_c * h_c)
            b_m, c_m, w_m, h_m = m_t.size()
            m_tf = m_t.reshape(b_m, c_m, w_m * h_m)
            b_x, c_x, w_x, h_x = x_t.size()
            x_tf = x_t.reshape(b_x, c_x, w_x * h_x)
            # b w c*h

            # accumulate_grad

            h_tf = self.encoder_model_1(h_tf)
            c_tf = self.encoder_model_2(c_tf)
            m_tf = self.encoder_model_3(m_tf)
            x_tf = self.encoder_model_4(x_tf)

            h_t = h_tf.reshape(b_h, c_h, w_h, h_h)
            c_t = c_tf.reshape(b_c, c_c, w_c, h_c)
            m_t = m_tf.reshape(b_m, c_m, w_m, h_m)
            x_t = x_tf.reshape(b_x, c_x, w_x, h_x)

        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = self.attn_sum_fussion(c_t, f_t, c_historys, c_historys) + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        if self.method==1:

            b_h, c_h, w_h, h_h = h_t.size()
            h_tf = h_t.reshape(b_h, c_h, w_h * h_h)  # b w c*h
            b_c, c_c, w_c, h_c = c_t.size()
            c_tf = c_t.reshape(b_c, c_c, w_c * h_c)
            b_m, c_m, w_m, h_m = m_t.size()
            m_tf = m_t.reshape(b_m, c_m, w_m * h_m)
            # b w c*h

            h_tf = self.encoder_model_1(h_tf)
            c_tf = self.encoder_model_2(c_tf)
            m_tf = self.encoder_model_3(m_tf)


            h_tf = h_tf.reshape(b_h, c_h, w_h, h_h)
            c_tf = c_tf.reshape(b_c, c_c, w_c, h_c)
            m_tf = m_tf.reshape(b_m, c_m, w_m, h_m)


            h_new = torch.cat((h_new,h_tf),dim=1)
            c_new = torch.cat((c_new, c_tf), dim=1)
            m_new = torch.cat((m_new, m_tf), dim=1)

            h_new = self.conv1x1_1(h_new)
            c_new = self.conv1x1_2(c_new)
            m_new = self.conv1x1_3(m_new)

        if self.method==2:
            b_h, c_h, w_h, h_h = h_new.size()
            h_tf = h_new.reshape(b_h, c_h, w_h * h_h)  # b w c*h
            b_c, c_c, w_c, h_c = c_new.size()
            c_tf = c_new.reshape(b_c, c_c, w_c * h_c)
            b_m, c_m, w_m, h_m = m_new.size()
            m_tf = m_new.reshape(b_m, c_m, w_m* h_m)

            h_tf = self.encoder_model_1(h_tf)
            c_tf = self.encoder_model_2(c_tf)
            m_tf = self.encoder_model_3(m_tf)

            h_new = h_tf.reshape(b_h, c_h, w_h, h_h)
            c_new = c_tf.reshape(b_c, c_c, w_c, h_c)
            m_new = m_tf.reshape(b_m, c_m, w_m, h_m)

        # print(h_new.shape)
        # print(c_new.shape)
        # print(m_new.shape)
        # print('**************************\n')

        return h_new, c_new, m_new









