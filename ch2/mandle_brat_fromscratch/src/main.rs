use num::complex::Complex;
use std::time::Instant;

fn main() {
    let console_height: i32 = 35;
    let console_width: i32 = 75;

    let x_one = -0.75;
    let x_two = -0.73;

    let y_one = 0.2;
    let y_two = 0.15;

    let now = Instant::now();
    let mut nums: Vec<Vec<i32>> = vec![vec![0; console_width as usize]; console_height as usize];

    for i in 0..console_height {
        for j in 0..console_width {
            let real = translate(i, 0, console_height, y_one, y_two);
            let imagine = translate(j, 0, console_width, x_one, x_two);
            let tested_num = test_number(
                Complex {
                    re: imagine,
                    im: real,
                },
                10000,
            );
            nums[i as usize][j as usize] = tested_num;
        }
    }

    for column in nums {
        for row in column {
            let val = match row {
                0..=2 => " ",
                3..=5 => ".",
                6..=10 => ",",
                11..=20 => "*",
                21..=50 => "o",
                51..=100 => "X",
                101..=250 => "%",
                251..=500 => "$",
                501..=5000 => "#",
                _ => "@",
            };
            print!("{}", val);
        }
        println!("");
    }
    println!("{:?}", Instant::now() - now);
}

fn test_number(number: Complex<f64>, times: i32) -> i32 {
    let previous = Complex { re: number.re, im: number.im };
    let mut current = Complex{ im: 0.0, re:0.0};

    for i in 0..times {
        current = current * current + previous;
        if current.norm() > 2.0 {
            return i;
        }
    }
    return times;
}
fn translate(
    val: i32,
    first_bound: i32,
    second_bound: i32,
    output_first: f64,
    output_second: f64,
) -> f64 {
    let normal_val = val - first_bound;
    let input_range = second_bound - first_bound;
    let percent_input = normal_val as f64 / input_range as f64;
    let output_range = output_second - output_first;
    let ret = (percent_input as f64 * output_range) + output_first;
    return ret;
}
