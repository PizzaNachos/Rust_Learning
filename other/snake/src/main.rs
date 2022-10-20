use device_query::{DeviceQuery, DeviceState, Keycode};
use std::{time, vec};

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

#[derive(Clone, Copy)]
enum Direction {
    North,
    South,
    East,
    West
}

struct SnakePeice {
    x:usize,
    y:usize,
    velocity:Direction,
    next: Box<Option<SnakePeice>>
}
impl SnakePeice{
    fn new(x:usize, y:usize) -> SnakePeice{
        SnakePeice { x: x, y: y, velocity: Direction::South, next: Box::new(Option::None) }
    }
    fn move_peice_and_children(&mut self){

        match &self.velocity{
            Direction::North => {self.x -= 1},
            Direction::South => {self.y += 1;},
            Direction::East => {self.y -= 1},
            Direction::West => {self.x += 1;}, 
        }
        self.x = if self.x == 0 {
            25
        } else if self.x == 25{
            1
        } else{
            self.x
        };

        self.y = if self.y == 0 {
            25
        } else if self.y == 25{
            1
        } else{
            self.y
        };


        match &mut (*(self.next)) {
            Some(child) => {
                child.move_peice_and_children();
                child.change_velocity(self.velocity);
            }
            None => ()
        }
    }

    fn change_velocity(&mut self, new_direction: Direction){
        self.velocity = new_direction;
    }

    fn get_vector_of_positions(&self) -> Vec<(usize,usize)>{
        let mut v = Vec::new();
        v.push((self.x,self.y));
        self.get_child_pos(v)
    }
    fn get_child_pos(&self, mut vector : Vec<(usize, usize)>) -> Vec<(usize,usize)>{
        print!("{}", self.x);
        match &(*self.next) {
            Some(n) => {
                vector.push((n.x,n.y));
                n.get_child_pos(vector)
            }
            None => vector
        }
    }
    fn add_child(&mut self){
        match &(*(self.next)) {
            Some(n) => n.add_child(),
            None => {
                *(self.next) = Option::Some(SnakePeice{
                    x: self.x,
                    y: self.y - 1,
                    velocity: self.velocity,
                    next: Box::new(Option::None)
                })
            }
        }


    }
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
    game_loop(board);
}


fn render_board(board : &Vec<Vec<Square>>){
    print!("\x1B[u");
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

fn game_loop(mut board : Vec<Vec<Square>>){
    let device_state = DeviceState::new();
    let mut _loop_delay = time::Duration::from_millis(66);
    let mut outside_key = Keycode::Space;
    let mut previous_loop_time = time::Instant::now();

    let mut snake = SnakePeice::new(10,10);
    *(snake.next) = Option::Some(SnakePeice::new(11,10));
    // *((*(snake.next)).unwrap().next) = Option::Some(SnakePeice::new(12,10));
    // *(snake.next) = Option::Some(SnakePeice::new(11,10));
    // *(snake.next) = Option::Some(SnakePeice::new(11,10));
    loop {
        let inside_key = device_state.get_keys();
        if !inside_key.is_empty() {
            outside_key = inside_key.last().unwrap_or(&outside_key).to_owned();
        }
        if time::Instant::now() - previous_loop_time >= _loop_delay {
            previous_loop_time = time::Instant::now();

            match &outside_key {
                Keycode::W => snake.change_velocity(Direction::North),
                Keycode::A => snake.change_velocity(Direction::East),
                Keycode::S => snake.change_velocity(Direction::West),
                Keycode::D => snake.change_velocity(Direction::South),
                _ => (),
            }
            snake.move_peice_and_children();
            board = put_snake_on_board(board, &snake);
            render_board(&board);

        }
    }
}

fn put_snake_on_board(mut board: Vec<Vec<Square>>, snake : &SnakePeice) -> Vec<Vec<Square>>{
    for (i,row) in board.iter_mut().enumerate(){
        for (j,space) in row.iter_mut().enumerate(){
            if i == 0 && j == 0 {
                *space = Square::Northwest;
                continue;
            } else 
            if i == 0 && j == 25 {
                *space = Square::Northeast;
                continue;
            } else 
            if i == 25 && j == 25 {
                *space  = Square::Southeast;
                continue;
            } else 
            if i == 25 && j == 25 {
                *space = Square::Southwest;
                continue;
            } else 
            if i == 0 {
                *space = Square::North;
                continue;
            } else 
            if i == 25 {
                *space = Square::South;
                continue;
            } else 
            if j == 25 {
                *space = Square::East;
                continue;
            } else 
            if j == 0 {
                *space = Square::West;
                continue;
            } else {
                *space = Square::Empty;
            }
        }
    }
    let pos = snake.get_vector_of_positions();
    println!("{:?}",pos);
    for (x,y) in pos{
        board[x][y] = Square::Snake
    }

    return board
}