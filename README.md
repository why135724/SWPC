# SWPC
Algo about self-supervised learning for MI

使用说明：在真实场景下，推荐使用CSP&FBCSP来判断当前是否处于静息态或运动想象态，EEGNet的性能或许更高，但是不稳定。而在明确当前为运动想象态的前提下，推荐使用正文中所说 EEGNET先supervised-learning再接self-supervised learning来分类，会取得更好的效果。

而EEGNET先SL再接SSL的方法在跨被试场景下会有明显更好的性能，可以与EA结合，但是在被试内场景下性能不够稳定。总结如下：CSP结合EEGNet效果更好，跨被试场景下EEGNET先SL再接SSL性能提升非常明显。

1.最基础的版本：被试内场景下的matlab代码的SWPC框架,请运行 asynchronous_fix_window_within.m  数据集放在 百度网盘： 链接：https://pan.baidu.com/s/14-f08DyaTVIuOKdsEY89zw 提取码：6uem

2.跨被试场景下的matlab代码的SWPC框架,请运行 asynchronous_fix_window_cross.m 预计11月1号更新

3.跨被试场景下的matlab代码的SWPC框架（结合EA）,请运行 asynchronous_fix_window_cross_EA.m 预计11月1号更新

4.跨被试或被试内场景python代码+supervised learning + self-supervised learning 预计11月更新完毕,coming soon...
