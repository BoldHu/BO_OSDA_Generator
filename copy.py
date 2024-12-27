'''
This origin train and test function in lstm
'''
# def train(epoches, model, iterator, optimizer, criterion, device):
#     # teaching forcing
#     epoch_loss = 0
#     train_loss_list = []
#     test_loss_list = []
#     for epoch in range(epoches):
#         model.train()
#         for i, (zeo, syn, label) in enumerate(iterator):
#             # concat the zeo and syn
#             conditional_synthesis = torch.cat([zeo, syn], dim=1)
#             conditional_synthesis, label = conditional_synthesis.to(device), label.to(device)
#             optimizer.zero_grad()
#             batch_size, seq_len = label.shape
            
#             # 这里有问题，应该是输入去掉最后一个token，输出去掉第一个token
#             # forward pass
#             output, _ = model(conditional_synthesis, label[:, :-1])
#             output_dim = output.shape[-1]
#             output = output.contiguous().view(-1, output_dim)
#             label = label[:, 1:].contiguous().view(-1)
#             # # forward pass
#             # output, _ = model(conditional_synthesis, label)
#             # output_dim = output.shape[-1] # vocab size
#             # # ignore the last token and reshape the output
#             # output = output[:, :-1, :].contiguous().view(-1, output_dim) # (batch_size * seq_len, vocab_size)
#             # # ignore the first token and reshape the label
#             # label = label[:, 1:].contiguous().view(-1) # (batch_size * seq_len)
            
#             # calculate the loss
#             loss = criterion(output, label)
#             loss.backward()
#             optimizer.step()
            
#             epoch_loss += loss.item()
#             # print information
#             if i % 100 == 0:
#                 print(f'Batch: {i}, Loss: {loss.item()}')
        
#         # calculate the average loss
#         train_loss_list.append(epoch_loss / len(iterator))
#         test_loss = test(model, test_dataloader, criterion, device)
#         test_loss_list.append(test_loss)
#         epoch_loss = 0
        
#         print(f'Epoch: {epoch}, Train Loss: {epoch_loss / len(iterator)}, Test Loss: {test_loss}')
#     return train_loss_list, test_loss_list

# def test(model, iterator, criterion, device):
#     model.eval()
#     total_loss = 0
#     generated_sequences = []
#     with torch.no_grad():
#         for i, (zeo, syn, label) in enumerate(iterator):
#             # concat the zeo and syn
#             conditional_synthesis = torch.cat([zeo, syn], dim=1)
#             conditional_synthesis, label = conditional_synthesis.to(device), label.to(device)
#             batch_size = conditional_synthesis.shape[0]
#             # set the initial hidden state
#             hidden = None
#             # set the initial input (batch_size, 1)
#             generated_sequence = label[:, :1] # (batch_size, 1)
#             batch_loss = 0
            
#             # auto-regressive
#             for t in range(label.shape[1]):
#                 # forward pass
#                 current_token = generated_sequence[:, -1].unsqueeze(1) # (batch_size, 1)
#                 output, hidden = model(conditional_synthesis, current_token, hidden) # (batch_size, 1, vocab_size)
                
#                 # sample for the next input
#                 next_token = sample_from_logits(output.squeeze(1)) # (batch_size,)
#                 generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(1)], dim=1) # (batch_size, t+2)
                
#                 # calculate the loss
#                 output_dim = output.shape[-1]
#                 output = output.view(-1, output_dim) # (batch_size, vocab_size)
#                 label_t = label[:, t]
#                 loss = criterion(output, label_t)
#                 batch_loss += loss.item()
            
#             batch_loss /= label.shape[1]
#             total_loss += batch_loss
#             generated_sequences.append(generated_sequence)
        
#         # calculate the average loss
#         total_loss /= len(iterator)
        
#     # convert the generated sequences to smiles
#     generated_sequences = torch.cat(generated_sequences, dim=0).cpu().numpy()
#     generated_sequences = sequence_to_smiles(generated_sequences, index_to_char)
#     print(generated_sequences[:10])
#     return total_loss