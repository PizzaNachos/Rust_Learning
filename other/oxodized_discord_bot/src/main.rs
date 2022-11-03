use ureq::get;
use regex::Regex;

use std::env;

use serenity::async_trait;
use serenity::model::channel::Message;
use serenity::model::gateway::Ready;
use serenity::prelude::*;

struct Handler;

struct Subreddit{
    name: String,
    last_accessed: usize
}
struct Subreddits{
    subreddits: Vec<Subreddit>,
    current_subreddit: usize
}


#[async_trait]
impl EventHandler for Handler {
    // Set a handler for the `message` event - so that whenever a new message
    // is received - the closure (or function) passed will be called.
    //
    // Event handlers are dispatched through a threadpool, and so multiple
    // events can be dispatched simultaneously.
    async fn message(&self, ctx: Context, msg: Message) {
        if msg.content == "!wow" {
            let links = fetch_links("").unwrap();
            // Sending a message can fail, due to a network error, an
            // authentication error, or lack of permissions to post in the
            // channel, so log to stdout when some error happens, with a
            // description of it.
            for link in links {
                if let Err(why) = msg.channel_id.say(&ctx.http, &link).await {
                    println!("Error sending message: {:?}", why);
                }
            }
        }
        if msg.content == "!dank" {
            let links = fetch_links("dankmemes").unwrap();
            // Sending a message can fail, due to a network error, an
            // authentication error, or lack of permissions to post in the
            // channel, so log to stdout when some error happens, with a
            // description of it.
            if let Err(why) = msg.channel_id.say(&ctx.http, &links[0]).await {
                println!("Error sending message: {:?}", why);
            }
            // for link in links {

            // }
        }
    }

    // Set a handler to be called on the `ready` event. This is called when a
    // shard is booted, and a READY payload is sent by Discord. This payload
    // contains data like the current user's guild Ids, current user data,
    // private channels, and more.
    //
    // In this case, just print what the current user's username is.
    async fn ready(&self, _: Context, ready: Ready) {
        println!("{} is connected!", ready.user.name);
    }
}

#[tokio::main]
async fn main() {
    // Configure the client with your Discord bot token in the environment.
    let token = String::from("");
    // Set gateway intents, which decides what events the bot will be notified about
    let intents = GatewayIntents::GUILD_MESSAGES
        | GatewayIntents::MESSAGE_CONTENT;

    // Create a new instance of the Client, logging in as a bot. This will
    // automatically prepend your bot token with "Bot ", which is a requirement
    // by Discord for bot users.
    let mut client =
        Client::builder(&token, intents).event_handler(Handler).await.expect("Err creating client");

    // Finally, start a single shard, and start listening to events.
    //
    // Shards will automatically attempt to reconnect, and will perform
    // exponential backoff until it reconnects.
    if let Err(why) = client.start().await {
        println!("Client error: {:?}", why);
    }
}


fn fetch_links(subreddit: &str) -> Result<Vec<String>, ureq::Error>{
    let reddit = String::from("https://old.reddit.com/r/") + subreddit;
    let body: String = get(&reddit)
        .set("Cookie", "over18=1")
        .call()?
        .into_string()?;

    let mut final_links = Vec::with_capacity(20);
    let re = Regex::new(r#"(?:data-url=")([^"]+)"#).unwrap();
    for cap in re.captures_iter(&body) {
        final_links.push(cap.get(1).unwrap().as_str().to_string());
    }
    final_links = final_links.into_iter().filter(|element| -> bool{ 
        !element.contains("redgifs")
    }).collect();
    println!("{} of links found", final_links.len());
    Ok(final_links)
}
