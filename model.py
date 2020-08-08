class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,size):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, size, 1, 1),
            nn.Dropout(0.2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=(2,2))
        )
        
        
        

        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        
        
        
        return x
      


class ConvBlock1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.Dropout(0.1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=(2,2))
        )
        
        
        

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        x = self.conv1(x)
        
        
        return x
      
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conva = nn.Sequential(
            ConvBlock1(in_channels=1, out_channels=16),
        ) 
        self.convb = nn.Sequential(
            ConvBlock(in_channels=16, out_channels=32,size=5),
        )
        self.convc = nn.Sequential(
            ConvBlock(in_channels=32, out_channels=32,size=3),
            
        )
       
        self.rnn = nn.GRU(3456, 128, 1,batch_first=True)
        self.fc =  nn.Linear(128, 3)
           
        
        self.sig = nn.Sigmoid()
      
        

    def forward(self, x):
        hidden = self.init_hidden()
        
        x = self.conva(x)
        
        x = self.convb(x)
        
        x = self.convc(x)
        
        x = self.convc(x)
       
        
        
        x, hidden = self.rnn(x.view(5,1,3456),hidden)
        
        x=x.reshape(5,128)
        
       
        x = self.fc(x)
        
        x=self.sig(x)
        return x
    
    def init_hidden(self):
        
        hidden = torch.zeros(1,5,128)
        hidden=hidden.cuda()
        return hidden

