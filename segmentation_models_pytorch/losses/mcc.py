import torch
from torch.nn.modules.loss import _Loss


class MCCLoss(_Loss):
    
    def __init__(
        self,
        eps: float = 1e-5
    ):  
        """Compute Matthews Correlation Coefficient Loss for image segmentation task.
        It only supports binary tasks

        Args:
            eps: Small epsilon to handle situations where all the samples in the dataset belong to one class

        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W)

        Reference
            https://github.com/kakumarabhishek/MCC-Loss

        """
        super(MCCLoss, self).__init__()
        self.eps = eps
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        """
        Args:
            y_pred: torch.Tensor of shape (N, C, H, W)
            y_true: torch.Tensor of shape (N, H, W) 
        
        Returns:
            loss: torch.Tensor
        """      

        assert y_true.size(0) == y_pred.size(0)

        bs = y_true.size(0)

        y_true = y_true.view(bs, 1, -1)
        y_pred = y_pred.view(bs, 1, -1)

        tp = torch.sum(torch.mul(y_pred, y_true)) + self.eps
        tn = torch.sum(torch.mul((1 - y_pred), (1 - y_true))) + self.eps
        fp = torch.sum(torch.mul(y_pred, (1 - y_true))) + self.eps
        fn = torch.sum(torch.mul((1 - y_pred), y_true)) + self.eps

        numerator = torch.mul(tp, tn) - torch.mul(fp, fn)
        denominator = torch.sqrt(
            torch.add(tp, 1, fp)
            * torch.add(tp, 1, fn)
            * torch.add(tn, 1, fp)
            * torch.add(tn, 1, fn)
        )
        mcc = torch.div(numerator.sum(), denominator.sum())
        loss = 1.0 - mcc

        return loss