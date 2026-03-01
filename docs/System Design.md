# BlogBoard System Design

1. In the backend operations, We have 5 jsons for Machine Learning, Deep Learning, Natural Language Processing, Computer Vision and General AI, which contains the date and its respective articles topics.
2. Based on those 5 jsons, we will generate a schedule.json where it will have a complete calendar of dates as keys and values as the articles topics, its respective domain.
3. The System will be refered to this schedule.json and that current date, so based on that current date it will fetch the article topic and its domain.
4. The system will generate the article in markdown format and save it in the `articles/{domain}` directory, also in that specific directory only we will have a articles.json in which contains the following data:
    * `id`: relative path of the article md file
    * `category`: category of the article
    * `title`: title of the article
    * `description`: description of the article
    * `date`: date of the article
    * `tags`: tags of the article
    * `readTime`: read time of the article (in minutes) [How to calculate read time is yet to be decided, any suggestions are welcome]
5. Now the website (JS) should fetch the articles from the `blogs/{domain}` directory and display them in the website dynamically.

## Blog Generation Logic
1. Based on the schedule.json, we will get the domain and topic.
2. Now we will generate a Cachy and SEO Friendly Title for the article.
3. Now we will generate a Cachy and SEO Friendly Short Description for the article.
4. Now we will generate a Cachy and SEO Friendly Tags for the article.
5. Now we will generate a Cachy and SEO Friendly Article for the article.
6. Generate a Final Markdown file with all the generated data.
7. Save the md file in the `blogs/{domain}` directory.
8. Update the `blogs.json` file in the `blogs/{domain}` directory.

Note: We need to do some github actions or Cloud something to do perform this blog generation everyday at a 09:00 AM IST as follows:

    - Monday: Machine Learning
    - Tuesday: Deep Learning
    - Wednesday: Statistics for AI
    - Thursday: Natural Language Processing
    - Friday: Computer Vision
    - Saturday: Generative AI