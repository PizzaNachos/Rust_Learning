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
            println!("In: {:032b}", instruction);
            println!("Opcode: {}, {:08b}", opcode, opcode);

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
            println!("Adding Byte {:08b}", *instruction);
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
        let t = (instruction >> 11) & 0b111;
        let rd = (instruction >> 7) & 0xf;
        let rs1 = (instruction >> 15) & 0xf;
        let imm = instruction >> 18;

        print!("t:{}", t);

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
    let add_ten_to_r1 : u32 = 0b00000001111_0000_000_0000_0010011;

    let add_something_to_31 : u32 = 0b01010000011_0001_000_1111_0010011;
    // add_ten_to_r1.to_ne_bytes().to_vec()
    match cpu.load_program(add_something_to_31.to_be_bytes().to_vec())
    {
        Ok(_) => {
            cpu.run();
        },
        Err(_) => println!("Program too large!")
    };
    
    println!("CPU Registers: ");
    for (i,r) in cpu.registers.iter().enumerate(){
        println!("R# {} = {:032b}",i, *r)

    }

    // cpu.current_operation = 0x8014;
    // cpu.registers[0] = 5;
    // cpu.registers[1] = 10;

    // cpu.run();

    // println!("Cpu Register 0: {}", cpu.registers[0]);
}
