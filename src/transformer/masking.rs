use candle_core::{Device, Result, Tensor, D};

pub fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

pub fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

pub fn create_seq_mask(q_len_valid: &[usize], k_len:usize, device: &Device) -> Result<Tensor> {
    let mut mask:Vec<u8>=Vec::new();
    for i in 0..q_len_valid.len(){
        let valid_len=q_len_valid[i];
        for j in 0..valid_len{
            mask.push(0);
        }
        for j in valid_len..k_len{
            mask.push(1);
        }
    }

    let t_mask=Tensor::from_slice(mask.as_slice(), &[q_len_valid.len(), k_len], device);
    return t_mask;
}

pub fn masked_softmax(attn:Tensor, mask:Option<&Tensor>)->Result<Tensor>{
    let shape=attn.dims();
    let masked_attn = match mask {
        None => attn,
        Some(m) => {
            let mask_expanded=m
            .unsqueeze(0)?
            .unsqueeze(0)?
            .repeat((shape[0], shape[1]))?;
            
            let ma=masked_fill(
            &attn,
            &mask_expanded,
            f32::NEG_INFINITY)?;
            ma
        }
    };
    let sm_masked_attn = candle_nn::ops::softmax(&masked_attn, D::Minus1);
    return sm_masked_attn;
}