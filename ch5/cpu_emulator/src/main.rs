struct RiscVCpu{
    program_counter: usize,
    registers: [u32;32],
    program_memory: [u8; 0x1000],
    global_memory: [u8; 0x1000],
}

impl RiscVCpu{
    fn read_instruction(&self) -> u32{
        let first = self.program_memory[self.program_counter] as u32 ;
        let second = first << 8 | self.program_memory[self.program_counter + 1] as u32 ;
        let third = second << 8 | self.program_memory[self.program_counter + 2] as u32 ;
        third << 8 | self.program_memory[self.program_counter + 3] as u32 
    }

    fn run(&mut self){
        loop{
            let instruction = self.read_instruction();
            if instruction == 0x0000 {
                break;
            };

            let opcode = instruction & 0x7f;
            println!("\nIn: {:032b}", instruction);
            println!("\tOpcode: {}, {:08b}", opcode, opcode);

            match opcode{
                0b00010011 => self.l_type(instruction),
                _ => println!("Un-supported Instruction {:08b}", opcode),
            }

            self.program_counter += 4;
        }

    }
    fn load_program(&mut self, program_bytes: Vec<u8>) -> Result<(), ()>{
        if program_bytes.len() > 0x1000 {
            return Err(())
        }
        for (i,instruction) in program_bytes.iter().enumerate(){
            // println!("Adding Byte {:08b}", *instruction);
            self.program_memory[i] = *instruction;
        }
        Ok(())
    }
    fn new() -> RiscVCpu{
        RiscVCpu{
            program_counter: 0,
            registers: [0; 32],
            program_memory: [0; 0x1000],
            global_memory: [0; 0x1000]
        }
    }

    fn l_type(&mut self, instruction: u32){
        let t = (instruction >> 12) & 0b111;
        let rd = (instruction >> 7) & 0x1f;
        let rs1 = (instruction >> 15) & 0x1f;
        let imm = instruction >> 20;

        // print!("t:{}", t);

        match t {
            // ADDI
            0b000 => {
                self.registers[rd as usize] = self.registers[rs1 as usize] + imm;
            }
            _ => println!("Not impliemted I type"),
        }
    }

}

fn main() {
    let mut cpu = RiscVCpu::new();
    let add_ten_to_r1 : u32 = 0b000001111_00000_000_00000_0010011;
    let add_something_to_31 : u32 = 0b010100011_00000_000_11111_0010011;

    // let i:u64 = 0b010100011_00000_000_11111_0010011_000001111_00000_000_00000_0010011;
    let program = read_ascii_file_to_vec("program.txt".to_string());
    for (i, byte) in program.iter().enumerate(){
        print!("{:08b} ", byte);
    }
    // let p = [add_ten_to_r1.to_be_bytes(), add_something_to_31.to_be_bytes()];
    // add_ten_to_r1.to_ne_bytes().to_vec()
    match cpu.load_program(program)
    {
        Ok(_) => {
            cpu.run();
        },
        Err(_) => println!("Program too large!")
    };
    
    println!("CPU Registers: ");
    for (i,r) in cpu.registers.iter().enumerate(){
        println!("R# {} = {:032b}={}",i, *r, *r)

    }

    // cpu.current_operation = 0x8014;
    // cpu.registers[0] = 5;
    // cpu.registers[1] = 10;

    // cpu.run();

    // println!("Cpu Register 0: {}", cpu.registers[0]);
}

fn read_ascii_file_to_vec(name: String) -> Vec<u8>{
    let mut r = vec![];

    if let Ok(file) = std::fs::read_to_string(name){
        let mut i = 0;
        let mut num : u8 = 0;
        for char in file.chars().into_iter(){
            match char{
                '0' => {
                    i += 1;
                    num = num << 1;
                },
                '1' => {
                    i += 1;
                    num = (num << 1) | 1;
                },
                _ => continue
            }

            if i % 8 == 0{
                r.push(num);
                i = 0;
                num = 0;
            }
        }
    };
    return r;
}
