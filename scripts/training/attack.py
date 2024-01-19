import torch
import torch.nn.functional as F

import pdb

def flag(model_forward, perturb_shape, y, args, optimizer, device, criterion) :
    model, forward = model_forward
    model.train()
    optimizer.zero_grad()

    perturb = torch.FloatTensor(*perturb_shape).uniform_(-args["step_size"], args["step_size"]).to(device)
    perturb.requires_grad_()
    out = forward(perturb)
    print(out.edata["pred"].dtype,y.dtype)
    loss = {}
    loss["loss"] = criterion(out.edata["pred"], y)
    # loss.requires_grad_(True)
    print("perturb:",perturb.grad,perturb[0])
    print("loss:",loss.grad,loss)
    loss /= args["m"]
    print("loss:", loss.grad, loss)
    print("perturb:",perturb.grad,perturb[0])

    
    for _ in range(args["m"]-1):
        loss.backward()
        perturb_data = perturb.detach() + args["step_size"] * torch.sign(perturb.grad.detach())
        perturb.data = perturb_data.data
        perturb.grad[:] = 0

        out = forward(perturb)
        loss = criterion(out.edata["pred"], y)
        loss /= args["m"]
        loss.backward()

    loss.backward()
    optimizer.step()

    return loss, g