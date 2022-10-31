// use std::

struct Matrix<T>{
    matrix: Vec<Vec<T>>,
    rows: usize,
    columns: usize,
}
impl Matrix<f64>{
    fn new(rows: usize, columns:usize) -> Matrix<f64>{
        Matrix { matrix: vec![vec![0.0;columns];rows], rows, columns}
    }
    fn add(&self, rhs: &Self) -> Self {
        let mut o = Matrix::new(self.rows, self.columns);
        for row in self.matrix.iter().enumerate(){
            for column in row.1.iter().enumerate(){
                o.matrix[row.0][column.0] = column.1 + rhs.matrix[row.0][column.0];
            }
        }
        return o;
    }
}

fn main() {
    let v1 = vec![vec![1.0,2.0,3.0],vec![4.0,5.0,6.0]];
    let v2 = vec![vec![0.1,0.2,0.3],vec![0.4,0.5,0.6]];
    let m2 = Matrix{matrix: v1, rows:2,columns:3};
    let m1 = Matrix{matrix: v2, rows:2,columns:3};

    let m3 = m1.add(&m2);
    println!("{:?},",m3.matrix); 
}
