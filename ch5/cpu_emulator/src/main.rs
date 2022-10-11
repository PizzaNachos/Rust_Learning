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
        let instruction = self.read_instruction();

        let opcode = instruction & 0x01111111;
        println!("{:032b}", opcode)
        // let c = ((opcode & 0xF000)>> 12) as u8;
        // let x = ((opcode & 0x0F00)>> 8) as u8;
        // let y = ((opcode & 0x00F0)>> 4) as u8;
        // let d = ((opcode & 0x000F)>> 0) as u8;

        // match (c,x,y,d) {
        //     (0x8, _, _, 0x4) => self.add_xy(x,y),
        //     _ => println!("Not Implimented")
        // }

    }

    // fn add_xy(&mut self, x:u8, y:u8){
    //     self.registers[x as usize] += self.registers[_y as usize];
    // }
}

fn main() {
    let mut cpu = RiscVCpu{
        program_counter: 0,
        registers: [0; 32],
        program_memory: [15; 0x1000],
        global_memory: [0; 0x1000]
    };


    // cpu.current_operation = 0x8014;
    // cpu.registers[0] = 5;
    // cpu.registers[1] = 10;

    cpu.run();

    // println!("Cpu Register 0: {}", cpu.registers[0]);
}
