#coding:utf-8
import tensorflow as tf
from tensorflow.python.layers import utils


def Adain(content,style,eps=1e-8):
    mean_c, var_c = tf.nn.moments(content,axes=[1,2],keep_dims= True)
    mean_s, var_s = tf.nn.moments(style,axes=[1,2],keep_dims=True)

    instance_normolization = (content -mean_c) / (var_c+eps)

    stylized_feature = instance_normolization*var_s+mean_s

    return stylized_feature


def wct_tf(content, style, alpha, eps=1e-8):
    # 去除掉 batch的维度，并获得图像的长宽和通道数
    content_t = tf.transpose(tf.squeeze(content), (2, 0, 1)) #交换维度
    style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

    Cc, Hc, Wc = tf.unstack(tf.shape(content_t)) #获取图像长宽，以及通道数
    Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

    # CxHxW -> CxH*W
    content_flat = tf.reshape(content_t, (Cc, Hc * Wc))
    style_flat = tf.reshape(style_t, (Cs, Hs * Ws))

    # 获得内容图像特征的协方差
    mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
    fc = content_flat - mc  #feature减去平均值
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc * Wc, tf.float32) - 1.) + tf.eye(Cc) * eps #计算协方差矩阵

    # 获得风格图像的协方差
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs * Ws, tf.float32) - 1.) + tf.eye(Cs) * eps #计算协方差矩阵

    # 奇异值分解，得到两个特征向量
    with tf.device('/cpu:0'):
        Sc, Uc, _ = tf.svd(fcfc)
        Ss, Us, _ = tf.svd(fsfs)


    # 去除掉过较小的值
    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    # zca白化，计算content feature的白化矩阵
    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))
    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:, :k_c], Dc), Uc[:, :k_c], transpose_b=True), fc)

    #对白化的content加上style图像的特征，使用zca白化
    Ds = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:, :k_s], Ds), Us[:, :k_s], transpose_b=True), fc_hat)

    # 加上style图像的平均值
    fcs_hat = fcs_hat + ms

    # 原图和混合图像所占的比例
    blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)

    # CxH*W -> CxHxW
    blended = tf.reshape(blended, (Cc, Hc, Wc))
    # CxHxW -> 1xHxWxC
    blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)

    return blended



### Style-Swap WCT ###

def wct_style_swap(content, style, alpha, patch_size=3, stride=1, eps=1e-8):
    '''
    加上style_swap 和wct
    '''
    content_t = tf.transpose(tf.squeeze(content), (2, 0, 1))
    style_t = tf.transpose(tf.squeeze(style), (2, 0, 1))

    Cc, Hc, Wc = tf.unstack(tf.shape(content_t))
    Cs, Hs, Ws = tf.unstack(tf.shape(style_t))

    # CxHxW -> CxH*W
    content_flat = tf.reshape(content_t, (Cc, Hc * Wc))
    style_flat = tf.reshape(style_t, (Cs, Hs * Ws))

    # Content 协方差
    mc = tf.reduce_mean(content_flat, axis=1, keep_dims=True)
    fc = content_flat - mc
    fcfc = tf.matmul(fc, fc, transpose_b=True) / (tf.cast(Hc * Wc, tf.float32) - 1.) + tf.eye(Cc) * eps

    # Style 协方差
    ms = tf.reduce_mean(style_flat, axis=1, keep_dims=True)
    fs = style_flat - ms
    fsfs = tf.matmul(fs, fs, transpose_b=True) / (tf.cast(Hs * Ws, tf.float32) - 1.) + tf.eye(Cs) * eps

    # 奇异分解
    with tf.device('/cpu:0'):
        Sc, Uc, _ = tf.svd(fcfc)
        Ss, Us, _ = tf.svd(fsfs)

    k_c = tf.reduce_sum(tf.cast(tf.greater(Sc, 1e-5), tf.int32))
    k_s = tf.reduce_sum(tf.cast(tf.greater(Ss, 1e-5), tf.int32))

    ### 白化content
    Dc = tf.diag(tf.pow(Sc[:k_c], -0.5))

    fc_hat = tf.matmul(tf.matmul(tf.matmul(Uc[:, :k_c], Dc), Uc[:, :k_c], transpose_b=True), fc)

    # reshape CxH*W -> 1xHxWxC
    whiten_content = tf.expand_dims(tf.transpose(tf.reshape(fc_hat, [Cc, Hc, Wc]), [1, 2, 0]), 0)

    ### 白化style
    Ds = tf.diag(tf.pow(Ss[:k_s], -0.5))
    whiten_style = tf.matmul(tf.matmul(tf.matmul(Us[:, :k_s], Ds), Us[:, :k_s], transpose_b=True), fs)
    # reshape CxH*W -> 1xHxWxC
    whiten_style = tf.expand_dims(tf.transpose(tf.reshape(whiten_style, [Cs, Hs, Ws]), [1, 2, 0]), 0)

    ### style_swap
    ss_feature = style_swap(whiten_content, whiten_style, patch_size, stride)
    # HxWxC -> CxH*W
    ss_feature = tf.transpose(tf.reshape(ss_feature, [Hc * Wc, Cc]), [1, 0])

    ###提取的style_swap加上style的颜色
    Ds_sq = tf.diag(tf.pow(Ss[:k_s], 0.5))
    fcs_hat = tf.matmul(tf.matmul(tf.matmul(Us[:, :k_s], Ds_sq), Us[:, :k_s], transpose_b=True), ss_feature)
    fcs_hat = fcs_hat + ms

    ### 所占比例
    blended = alpha * fcs_hat + (1 - alpha) * (fc + mc)
    # CxH*W -> CxHxW
    blended = tf.reshape(blended, (Cc, Hc, Wc))
    # CxHxW -> 1xHxWxC
    blended = tf.expand_dims(tf.transpose(blended, (1, 2, 0)), 0)

    return blended


def style_swap(content, style, patch_size, stride):

    nC = tf.shape(style)[-1]  # 特征图像通道
    #print(patch_size)
    #print(content.shape)

    ### 从style feature中提取一块图像
    style_patches = tf.extract_image_patches(style, [1, patch_size, patch_size, 1], [1, stride, stride, 1],
                                             [1, 1, 1, 1], 'VALID')
    #print(style_patches.shape)
    before_reshape = tf.shape(style_patches)  # NxRowsxColsxPatch_size*Patch_size*nC
    style_patches = tf.reshape(style_patches, [before_reshape[1] * before_reshape[2], patch_size, patch_size, nC])
    #print(style_patches.shape)
    style_patches = tf.transpose(style_patches, [1, 2, 3, 0])  # Patch_sizexPatch_sizexIn_CxOut_c
    #print(style_patches.shape)

    # l2泛化
    style_patches_norm = tf.nn.l2_normalize(style_patches, dim=3)

    #每一个style_path与原图像相乘，得到一个相乘的结果为一个通道
    ss_enc = tf.nn.conv2d(content,
                          style_patches_norm,
                          [1, stride, stride, 1],
                          'VALID')

    # 在每个通道内找到最大的值，即为style提取patch和content最接近的区域
    #print(ss_enc.shape)
    ss_argmax = tf.argmax(ss_enc, axis=3)
    #print(ss_argmax.shape)
    encC = tf.shape(ss_enc)[-1]

    # 每一个patch, 标记最大那个区域为1，其他区域为0
    ss_oh = tf.one_hot(ss_argmax, encC, 1., 0., 3)
    #print(ss_oh.shape)

    # 输出图像的大小
    deconv_out_H = utils.deconv_output_length(tf.shape(ss_oh)[1], patch_size, 'valid', stride)
    deconv_out_W = utils.deconv_output_length(tf.shape(ss_oh)[2], patch_size, 'valid', stride)
    deconv_out_shape = tf.stack([1, deconv_out_H, deconv_out_W, nC])

    # 反卷积，还原出来的大小为原图大小，但是只有最相近的patch的信息
    ss_dec = tf.nn.conv2d_transpose(ss_oh,
                                    style_patches,
                                    deconv_out_shape,
                                    [1, stride, stride, 1],
                                    'VALID')

    ### 重叠部分求平均指
    ss_oh_sum = tf.reduce_sum(ss_oh, axis=3, keep_dims=True)

    filter_ones = tf.ones([patch_size, patch_size, 1, 1], dtype=tf.float32)
    #print(filter_ones.shape)

    deconv_out_shape = tf.stack([1, deconv_out_H, deconv_out_W, 1])

    counting = tf.nn.conv2d_transpose(ss_oh_sum,
                                      filter_ones,
                                      deconv_out_shape,
                                      [1, stride, stride, 1],
                                      'VALID')

    counting = tf.tile(counting, [1, 1, 1, nC])

    interpolated_dec = tf.divide(ss_dec, counting)

    return interpolated_dec
