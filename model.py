import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch

class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, stride=1):
        self.bias_opening = False
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=self.bias_opening,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias_opening,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != mid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    mid_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=self.bias_opening,
                ),
                nn.BatchNorm2d(mid_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockVerse(nn.Module):
    def __init__(self, in_channels, mid_channels, stride):
        self.bias_opening = False
        super(BasicBlockVerse, self).__init__()
        self.conv1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            output_padding=stride - 1,
            bias=self.bias_opening,
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=self.bias_opening,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != mid_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels,
                    mid_channels,
                    kernel_size=1,
                    stride=stride,
                    output_padding=stride - 1,
                    bias=self.bias_opening,
                ),
                nn.BatchNorm2d(mid_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, block_verse, in_channels=None):
        super(ResNetEncoder, self).__init__()
        self.in_channels = in_channels
        self.bn1 = nn.BatchNorm2d(self.in_channels)

        self.conv1 = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = block(in_channels=self.in_channels, mid_channels=4, stride=1)
        self.layer2 = block(in_channels=4, mid_channels=8, stride=2)
        self.layer3 = block(in_channels=8, mid_channels=16, stride=2)
        self.layer4 = block(in_channels=16, mid_channels=32, stride=2)
        self.layer1_ = block_verse(in_channels=32, mid_channels=16, stride=2)
        self.layer2_ = block_verse(in_channels=16, mid_channels=8, stride=2)
        self.layer3_ = block_verse(in_channels=8, mid_channels=4, stride=2)

        self.conv2 = nn.Conv2d(in_channels=4, out_channels=4, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer1_(out)
        out = self.layer2_(out)
        out = self.layer3_(out)
        out = self.conv2(out)
        return out


class ResNetDecoderA(nn.Module):
    def __init__(self, block, in_channels=None):
        super(ResNetDecoderA, self).__init__()
        self.in_channels = in_channels
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.sigmoid = nn.Sigmoid()

        self.layer1 = block(in_channels=self.in_channels, mid_channels=2, stride=1)
        self.layer2 = block(in_channels=2, mid_channels=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.conv2(out)
        return out


class ResNetDecoderB(nn.Module):
    def __init__(self, block, in_channels=None):
        super(ResNetDecoderB, self).__init__()
        self.in_channels = in_channels
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.sigmoid = nn.Sigmoid()

        self.layer1 = block(in_channels=self.in_channels, mid_channels=2, stride=1)
        self.layer2 = block(in_channels=2, mid_channels=1, stride=1)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.conv2(out)
        return out


def resnet_encoder(in_channels=28):
    return ResNetEncoder(BasicBlock, BasicBlockVerse, in_channels=in_channels)


def resnet_decoder_a(in_channels=4):
    return ResNetDecoderA(BasicBlock, in_channels=in_channels)


def resnet_decoder_b(in_channels=5):
    return ResNetDecoderB(BasicBlock, in_channels=in_channels)


class GAT(torch.nn.Module):
    def __init__(self, attribute_num=1, num_nodes=33):
        super(GAT, self).__init__()
        self.num_nodes = num_nodes
        self.conv1 = GATConv(attribute_num, 64, heads=4, dropout=0.3)
        self.conv2 = GATConv(64 * 4, 32, heads=4, concat=False, dropout=0.3)
        # self.conv3 = GATConv(16 * 4, 2, heads=2, concat=False, dropout=0.3)
        self.linear1 = nn.Linear(1056, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # re-grouping according to batch index
        batch_size = batch.unique().size(0)
        # divide into the node features of each graph
        x_split = torch.split(x, self.num_nodes, dim=0)
        x_grouped = torch.stack(x_split, dim=0)  # [batch_size, num_nodes, hidden_dim]
        x_flattened = x_grouped.view(batch_size, -1)  # [batch_size, num_nodes * hidden_dim]

        # print(x_flattened.shape)
        x = F.relu(self.linear1(x_flattened))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        x = torch.sigmoid(x)
        return x

class DQN_Agent_Transformer_GAT_PRE(nn.Module):
    def __init__(self, state_dim, num_rooms, action_dim, d_model=64, gat_hidden_dim=32, control_dim=None):
        super(DQN_Agent_Transformer_GAT_PRE, self).__init__()
        self.state_dim = state_dim
        self.num_rooms = num_rooms
        self.control_dim = control_dim if control_dim is not None else num_rooms + 3
        self.action_dim = action_dim
        self.gat_hidden_dim = gat_hidden_dim

        self.state_norm = nn.LayerNorm(state_dim)  # state normalization
        self.embedding = nn.Linear(state_dim, d_model) # embedding layer

        #### encoding layout result
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # single graph attention layer
        self.gat_conv1 = GATConv(1, gat_hidden_dim, heads=2, concat=True, dropout=0.2)
        self.gat_conv2 = GATConv(gat_hidden_dim * 2, gat_hidden_dim, heads=1, concat=True, dropout=0.2)

        #### encoding adjacent matrix
        # each row is a token, mapped to d_model dimensions
        self.embedding_graph = nn.Linear(self.control_dim, d_model)

        # define Transformer encoder layer (explicitly set batch_first=True)
        encoder_layer_graph = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.transformer_encoder_graph = nn.TransformerEncoder(encoder_layer_graph, num_layers=3)

        # linear layer to aggregate room information
        self.mlp_decoder = nn.Sequential(
            nn.Linear(1984, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
        )

        # linear layers to output values
        self.fc_value = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, input):
        # get data from the input dictionary
        state = input['state']
        mask = input['mask']
        graph_data = input['controls_array']
        
        batch_size = state.shape[0]
        
        # get valid actions
        state_pack_reshaped = state.view(batch_size, self.action_dim, self.num_rooms, self.state_dim)
        batch_indices, action_indices = mask.nonzero(as_tuple=True)
        valid_states = state_pack_reshaped[batch_indices, action_indices, :, :]  # (num_valid, num_rooms, state_dim)

        # Transformer encoding
        valid_states = self.state_norm(valid_states)
        x = self.embedding(valid_states)
        x = self.transformer_encoder(x)
        x = x.view(x.size(0), -1)  # (num_valid, num_rooms * d_model)

        # adjacent matrix encoding
        # print(type(graph_data))
        gat_out = self.embedding_graph(graph_data)
        gat_out = self.transformer_encoder_graph(gat_out)
        # pick the last row as the graph representation
        graph_rep = gat_out[:, -1, :]  # [batch_size, d_model]

        x_graph = graph_rep.view(batch_size, -1)    # (batch_size, control_dim * gat_hidden_dim)
        # pick all valid actions' graph representation
        x_graph = x_graph.unsqueeze(1).expand(-1, self.action_dim, -1).contiguous()
        x_graph = x_graph[batch_indices, action_indices, :]  # (num_valid, control_dim * gat_hidden_dim)

        # combine layout and graph features
        x_cat = torch.cat([x, x_graph], dim=1)  # (num_valid, num_rooms * d_model + control_dim * gat_hidden_dim)
        x = self.mlp_decoder(x_cat)
        values = self.fc_value(x).squeeze(-1)

        # pad back to original shape
        padded_values = torch.full((batch_size, self.action_dim), -1e9, device=state.device)
        padded_values[batch_indices, action_indices] = values

        return padded_values

