use clap::{App, Arg};
use regex::Regex;
use std::fs::File;
use std::io;
use std::io::prelude::*;
use std::io::BufReader;

fn main() {
    let args = App::new("grep_lite")
        .version("0.0.1")
        .about("searches for patterns")
        .arg(
            Arg::with_name("pattern")
                .help("The pattern to search for")
                .takes_value(true)
                .required(true),
        )
        .arg(
            Arg::with_name("input")
                .help("File to search")
                .takes_value(true)
                .required(false),
        )
        .arg(
            Arg::with_name("context")
                .help("Number of lines to print")
                .takes_value(false)
                .required(false),
        )
        .get_matches();

    let pattern = args.value_of("pattern").unwrap();
    let re = Regex::new(pattern).unwrap();

    let input = args.value_of("input").unwrap_or("-");

    let default_context = String::from("0");
    let context = args
        .get_one::<String>("context")
        .unwrap_or(&default_context)
        .parse()
        .unwrap();
        
    if input == "-" {
        let stdin = io::stdin();
        let reader = stdin.lock();
        proscess_lines(reader, re, context);
    } else {
        let f = File::open(input).unwrap();
        let reader = BufReader::new(f);
        proscess_lines(reader, re, context);
    }
}

fn proscess_lines<T: BufRead + Sized>(reader: T, re: Regex, context: usize) {
    let mut current_window = 0;
    for (i, line_) in reader.lines().enumerate() {
        let line = line_.unwrap();
        match re.find(&line) {
            Some(_) => {
                print!("{}  ", i);
                for word in line.split(" ").collect::<Vec<&str>>() {
                    if re.is_match(word){
                        print!("\x1b[91m{}\x1b[0m ",word);
                    } else {
                        print!("{} ",word);
                    }
                }
                println!("");
                current_window = context;
            }
            None => {
                if current_window > 0 {
                    println!("{}:  {}",i, line);
                    current_window -= 1;
                }
            }
        }
    }
}
