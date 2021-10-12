#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch


# In[2]:


t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]
t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4]
t_c = torch.tensor(t_c)#pytoh数组转成pytorch-tensor
t_u = torch.tensor(t_u)


# In[3]:


import matplotlib.pyplot as plt

plt.figure(figsize=(18,12))
plt.scatter(t_u, t_c,color='b') ##可以用来观察存在线型的关系
plt.xlabel("measurement", fontsize=13)
plt.ylabel("temperature", fontsize=13)

plt.show()


# In[4]:


def model(t_u, w, b):
    return w * t_u + b

def loss_fn(t_p, t_c):
    squared_diffs = (t_p - t_c)**2
    return squared_diffs.mean()


# In[5]:


w = torch.ones(())
b = torch.zeros(())

t_p = model(t_u, w, b)
t_p


# In[6]:


loss = loss_fn(t_p,t_c)
loss


# In[7]:


# 数值梯度计算
delta = 0.1
loss_rate_of_change_w =     (loss_fn(model(t_u, w + delta, b), t_c) -
    loss_fn(model(t_u, w - delta, b), t_c)) / (2.0 * delta)


# In[8]:


learning_rate = 1e-2

w = w - learning_rate * loss_rate_of_change_w


# In[9]:


loss_rate_of_change_b =     (loss_fn(model(t_u, w, b + delta), t_c) -
    loss_fn(model(t_u, w, b - delta), t_c)) / (2.0 * delta)

b = b - learning_rate * loss_rate_of_change_b


# In[10]:


# 梯度下降解析解
def dloss_fn(t_p, t_c):
    dsq_diffs=2* (t_p - t_c) / t_p.size(0)
    return dsq_diffs

def dmodel_dw(t_u, w, b):
    return t_u

def dmodel_db(t_u, w, b):
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])


# In[11]:


# 定义每次train的迭代
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c) #forward
        grad = grad_fn(t_u, t_c, t_p, w, b) #backward
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print(f'       Params: ',params)
        print(f'       Grad: ',grad)
    
    return params


# In[12]:


# 观察blow up现象
training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)


# In[13]:


# 调小learning_rate
training_loop(
    n_epochs = 100,
    learning_rate = 1e-4,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_u,
    t_c = t_c)


# - Broadcasting广播机制

# In[14]:


x = torch.ones(())
y = torch.ones(3,1)
z = torch.ones(1,3)
a = torch.ones(2, 1, 1)
print(f"shapes: x: {x.shape}, y: {y.shape}")
print(f" z: {z.shape}, a: {a.shape}")
print("x * y:", (x * y).shape)
print("y * z:", (y * z).shape)
print("y * z * a:", (y * z * a).shape)


# - end

# In[15]:


# 5.4.4 Normalizing inputs输入标准化处理
t_un = 0.1 * t_u


# In[16]:


training_loop(
    n_epochs = 100,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c)


# In[17]:


# 增大epoch数
params = training_loop(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0]),
    t_u = t_un,
    t_c = t_c,
   )


# In[18]:


# 可视化
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt

t_p = model(t_un, *params)
fig = plt.figure(dpi=300)
plt.xlabel("Temperature (°Fahrenheit)")
plt.ylabel("Temperature (°Celsius)")
plt.plot(t_u.numpy(), t_p.detach().numpy())
plt.plot(t_u.numpy(), t_c.numpy(), 'o')


# In[19]:


# 5.5 PyTorch’s autograd: Backpropagating all things反向传播全过程

#def model(t_u, w, b):
 #   return w * t_u + b

#def loss_fn(t_p, t_c):
 #   squared_diffs = (t_p - t_c)**2
  #  return squared_diffs.mean()


# In[20]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
# 查看参数梯度是否为none
params.grad is None


# In[21]:


loss = loss_fn(model(t_u, *params), t_c)
loss.backward()

params.grad


# In[22]:


params.grad is None


# In[23]:


# 清空参数梯度为接下来反向传播准备
if params.grad is not None:
    params.grad.zero_()


# In[24]:


params.grad


# In[25]:


def training_loop_without_parmas_output(n_epochs, learning_rate, params, t_u, t_c):
    
    for epoch in range(1, n_epochs + 1):
        w, b = params
        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c) #forward
        grad = grad_fn(t_u, t_c, t_p, w, b) #backward
        params = params - learning_rate * grad
        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        #print(f'       Params: ',params)
        #print(f'       Grad: ',grad)
    
    return params


# In[26]:


training_loop_without_parmas_output(
    n_epochs = 5000,
    learning_rate = 1e-2,
    params = torch.tensor([1.0, 0.0], requires_grad=True),
    t_u = t_un,
    t_c = t_c)


# In[27]:


import torch.optim as optim

# 优化器菜单
dir(optim)


# In[28]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-5
# SGD随机梯度下降
optimizer = optim.SGD([params], lr=learning_rate)


# In[29]:


t_p = model(t_u, *params)
loss = loss_fn(t_p, t_c) #正向传播
loss.backward() #反向传播
optimizer.step() #一步更新所有参数

params


# In[30]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

params


# In[31]:


def training_loop_2(n_epochs, optimizer, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
    
    return params


# In[32]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate2 = 1e-2
optimizer = optim.SGD([params], lr=learning_rate)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer.zero_grad()
loss.backward()
optimizer.step()

params


# In[33]:


training_loop_2(
    n_epochs = 5000,
    optimizer = optimizer,
    params = params,
    t_u = t_un,
    t_c = t_c)


# In[34]:


params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-1
optimizer = optim.Adam([params], lr=learning_rate)

training_loop_2(
    n_epochs = 2000,
    optimizer = optimizer,
    params = params,
    t_u = t_u,
    t_c = t_c)


# In[35]:


# 5.5.3 SPLITTING A DATASET
n_samples = t_u.shape[0]
n_val = int(0.2 * n_samples)

shuffled_indices = torch.randperm(n_samples)
train_indices = shuffled_indices[:-n_val]

val_indices = shuffled_indices[-n_val:]
train_indices, val_indices

# Since these are random, don’t be surprised if your values end up different from here on out.


# In[36]:


# 训练集
train_t_u = t_u[train_indices]
train_t_c = t_c[train_indices]
# 验证集
val_t_u = t_u[val_indices]
val_t_c = t_c[val_indices]
#输入标准化处理
train_t_un = 0.1 * train_t_u
val_t_un = 0.1 * val_t_u


# In[37]:


def training_loop_3(n_epochs, optimizer, params, train_t_u, val_t_u,train_t_c, val_t_c):
    
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)
        optimizer.zero_grad()
        train_loss.backward() #Note that there is no val_loss.backward()here, 
                                #since we don’t want to train themodel on the validation data.
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print(f"Epoch {epoch}, Training loss {train_loss.item():.4f},"
                    f" Validation loss {val_loss.item():.4f}")
        
    return params


# In[38]:


params_temp_1 = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate_temp_1 = 1e-2
optimizer_temp_1 = optim.SGD([params_temp_1], lr=learning_rate_temp_1)

t_p = model(t_un, *params)
loss = loss_fn(t_p, t_c)

optimizer_temp_1.zero_grad()
loss.backward()
optimizer.step()


# In[39]:


training_loop_3(
    n_epochs = 3000,
    optimizer = optimizer_temp_1,
    params = params_temp_1,
    train_t_u = train_t_un,
    val_t_u = val_t_un,
    train_t_c = train_t_c,
    val_t_c = val_t_c)


# In[40]:


# Autograd nits and switching it off
def training_loop_4(n_epochs, optimizer, params, train_t_u, val_t_u,train_t_c, val_t_c):
    
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)
    # switch off autograd when we don’tneed it, using the torch.no_grad context manager.
    with torch.no_grad():
        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)
        assert val_loss.requires_grad == False
        
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


# In[41]:


# is_train bool值控制autograd是否使能
def calc_forward(t_u, t_c, is_train):
    with torch.set_grad_enabled(is_train):
        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
    return loss

