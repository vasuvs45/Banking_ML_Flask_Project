import matplotlib.pyplot as plt
import io


# def generate_pie_plot(predictions):
#     plt.figure(figsize=(4, 4))
#     num_yes = sum(predictions)
#     num_no = len(predictions) - num_yes

#     labels = ['Yes', 'No']
#     sizes = [num_yes, num_no]
#     colors = ['lightcoral', 'lightskyblue']
#     explode = (0.1, 0)

#     plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
#     plt.axis('equal')
#     plt.title("Customer Subscription Prediction")

#     buffer = io.BytesIO()
#     plt.savefig(buffer, format='png')
#     buffer.seek(0)
#     plt.close()
#     return buffer

def generate_bar_plot(predictions):
    num_yes = sum(predictions)  
    num_no = len(predictions) - num_yes  

    labels = ['Yes', 'No']
    counts = [num_yes, num_no]

    plt.figure(figsize=(6, 6))
    plt.bar(labels, counts, color=['lightcoral', 'lightskyblue'])
    plt.xlabel('Prediction')
    plt.ylabel('Count')
    plt.title('Customer Subscription Prediction')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer