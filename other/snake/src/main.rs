use device_query::{DeviceQuery, DeviceState, Keycode};
use std::{time};

enum Square {
    Empty,
    Apple,
    Snake,
    North,
    South,
    East,
    West,
    Northwest,
    Northeast,
    Southwest,
    Southeast
}
enum Direction {
    North,
    South,
    East,
    West
}

struct SnakePeice{
    x:usize,
    y:usize,
    velocity:Direction,
    next: Box<Option<SnakePeice>>
}

fn main() {
    let width:usize = 25;
    let height:usize = 25;
    let mut board : Vec<Vec<Square>> = vec![vec![]];
    for i in 0..=height{
        board.push(vec![]);
        for j in 0 ..=width{
            if i == 0 && j == 0 {
                board[i].push(Square::Northwest);
                continue;
            }
            if i == 0 && j == width {
                board[i].push(Square::Northeast);
                continue;
            }
            if i == height && j == width {
                board[i].push(Square::Southeast);
                continue;
            }
            if i == height && j == 0 {
                board[i].push(Square::Southwest);
                continue;
            }
            if i == 0 {
                board[i].push(Square::North);
                continue;
            }
            if i == width {
                board[i].push(Square::South);
                continue;
            }
            if j == height {
                board[i].push(Square::East);
                continue;
            }
            if j == 0 {
                board[i].push(Square::West);
                continue;
            }
            board[i].push(Square::Empty);
        }
    }
    render_board(&board);
    game_loop(board);
}


fn render_board(board : &Vec<Vec<Square>>){
    for row in board {
        for space in row{
            match space {
                Square::Empty => print!(" "),
                Square::Apple => print!("@"),
                Square::Snake => print!("S"), 
                Square::North => print!("\u{2550}"),
                Square::South => print!("\u{2550}"),
                Square::East =>  print!("\u{2551}"),
                Square::West =>  print!("\u{2551}"),
                Square::Northwest => print!("\u{2554}"),
                Square::Northeast => print!("\u{2557}"),
                Square::Southwest => print!("\u{255A}"),
                Square::Southeast => print!("\u{255D}"),
            }
        }
        println!("");
    }
}

fn game_loop(_board : Vec<Vec<Square>>){
    let device_state = DeviceState::new();
    let mut _loop_delay = time::Duration::from_millis(1000);
    let mut outside_key = Keycode::Space;
    let mut previous_loop_time = time::Instant::now();
    loop {
        let inside_key = device_state.get_keys();
        if !inside_key.is_empty() {
            outside_key = inside_key.last().unwrap_or(&outside_key).to_owned();
        }
        if time::Instant::now() - previous_loop_time >= _loop_delay {
            previous_loop_time = time::Instant::now();

            match &outside_key {
                Keycode::W => println!("w"),
                Keycode::A => println!("a"),
                Keycode::S => println!("s"),
                Keycode::D => println!("d"),
                _ => (),
            }
        }
    }
}

fn move_snake(&mut snake : SnakePeice){
    while snake.next.is_some() {
        match snake.velocity{
            Direction::North => {
                snake.x -= 1;
            },
            Direction::West => {
                snake.y -= 1;
            },
            Direction::East => {
                snake.y += 1;
            },
            Direction::South => {
                snake.x += 1;
            }
        }
        match *snake.next {
            Some(mut next_snake) => {
                next_snake.velocity = snake.velocity;
                snake = next_snake;
            },
            None => ()
        }
    }
}

fn put_snake_on_board(mut board: Vec<Vec<Square>>, snake : &SnakePeice) -> Vec<Vec<Square>>{
    for (i,row) in board.iter_mut().enumerate(){
        for (j,space) in row.iter_mut().enumerate(){
            if i == 0 && j == 0 {
                board[i][j] = (Square::Northwest);
                continue;
            }
            if i == 0 && j == row.len() {
                board[i].push(Square::Northeast);
                continue;
            }
            if i == board.len() && j == row.len() {
                board[i].push(Square::Southeast);
                continue;
            }
            if i == board.len() && j == row.len() {
                board[i].push(Square::Southwest);
                continue;
            }
            if i == 0 {
                board[i].push(Square::North);
                continue;
            }
            if i == board.len() {
                board[i].push(Square::South);
                continue;
            }
            if j == row.len() {
                board[i].push(Square::East);
                continue;
            }
            if j == 0 {
                board[i].push(Square::West);
                continue;
            }
            board[i].push(Square::Empty);
        }
    }
    while(snake.next.is_some()){
        board[snake.x][snake.y] = Square::Snake;
    }
    for (i,row) in board.iter_mut().enumerate(){
        for (j,space) in row.iter_mut().enumerate(){

        }
    }


    return board
}
// fn move_snake_peice(mut snake : &mut SnakePeice){
//     if snake.next.is_none() {
//         return;
//     }



// }