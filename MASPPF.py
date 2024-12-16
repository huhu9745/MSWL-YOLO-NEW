class MASPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.
        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.cv3 = Conv(c_ * 2, c_, 1, 1)
        self.cv4 = Conv(c_ * 4, 512, 1, 1)
        self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.avg_pool = nn.AvgPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through SPPF layer."""
        # 第一个卷积512
        # print(f"After cv1: {x.shape}")
        x = self.cv1(x)
        # print(f"After cv1: {x.shape}")
        # 进行并行池化
        max_pooled = self.max_pool(x)
        # print(f"After max: {max_pooled.shape}")
        avg_pooled = self.avg_pool(x)
        # print(f"After max: {avg_pooled.shape}")
        y1 = torch.cat((max_pooled,avg_pooled),1)
        y2 = self.cv3(y1)
        # print(f"After first pool and cv3: {y2.shape}")
        max_pooled2 = self.max_pool(y2)
        # print(f"After max2: {max_pooled2.shape}")
        avg_pooled2 = self.avg_pool(y2)
        # print(f"After avg2: {avg_pooled2.shape}")
        y3 = torch.cat((max_pooled2,avg_pooled2),1)  #256
        y4 = self.cv3(y3)
        # print(f"After second pool and cv3: {y4.shape}") #256
        max_pooled3 = self.max_pool(y4)
        # print(f"After max3: {max_pooled3.shape}")
        avg_pooled3 = self.avg_pool(y4)
        # print(f"After avg3: {max_pooled3.shape}")
        y5 = torch.cat((max_pooled3,avg_pooled3),1)
        # print(f"After third pool and cv3: {y5.shape}")
        y6 = self.cv3(y5)
        # print(f"y6: {y6.shape}")
        y7 = torch.cat((x,y2,y4,y6),1)
        # print(f"After third pool and cv3: {y7.shape}")
        y8 = self.cv4(y7)
        # print(f"y8888888 {y8.shape}")

        return y8


