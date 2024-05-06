use rustygrad::*;

fn main() {

    let xs = [
       vals(vec![2.0, 3.0, -1.0]),
       vals(vec![3.0, -1.0, 0.5]),
       vals(vec![0.5, 1.0, 1.0]),
       vals(vec![1.0, 1.0, -1.0])
    ];

    let ys = vals(vec![1.0, -1.0, -1.0, 1.0]);
        
    let nn = Mlp::new(3, vec![4, 1]);
    
    for _ in 0..50 {
        let preds: Vec<Val> = xs
            .iter()
            .map(|x| nn.forward(x.to_vec())[0].clone())
            .collect();

        let losses: Vec<Val> = ys
            .iter()
            .zip(&preds)
            .map(|(y, pred)| (y - pred).pow(2.))
            .collect();

        let size = val(losses.len() as f64);
        let loss = &losses.into_iter().sum::<Val>() / &size;

        loss.back();

        for parameter in &nn.parameters() {
            *parameter.data_mut() -= 0.1 * *parameter.grad()
        }

        nn.zero_grad();

        println!("loss: {loss}, ");
    }

    let preds: Vec<f64> = xs
          .iter()
          .map(|x| *nn.forward(x.to_vec())[0].clone().data())
          .collect();

    println!("{preds:?}");

}

